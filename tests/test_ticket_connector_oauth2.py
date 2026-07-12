from __future__ import annotations

import asyncio
from collections.abc import Generator
from http import HTTPStatus
from time import monotonic
from types import TracebackType
from unittest import mock

import pytest

from discord_bot.configuration import Config, Singleton
from discord_bot.helpers import ticket_connector
from discord_bot.helpers.ticket_connector import TicketOrder


class _FakeResponse:
    def __init__(self, status: int, json_payload: object = None, text_payload: str = "") -> None:
        self.status = status
        self._json_payload = {} if json_payload is None else json_payload
        self._text_payload = text_payload

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    async def json(self) -> object:
        await asyncio.sleep(0)
        return self._json_payload

    async def text(self) -> str:
        await asyncio.sleep(0)
        return self._text_payload


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.post_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.post_calls.append({"url": url, "kwargs": kwargs})
        return self.response

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.get_calls.append({"url": url, "kwargs": kwargs})
        return self.response


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Reset singleton instances and oauth env vars between tests."""
    monkeypatch.delenv("TICKETS_OAUTH2_CLIENT_ID", raising=False)
    monkeypatch.delenv("TICKETS_OAUTH2_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("TICKETS_OAUTH2_TOKEN_URL", raising=False)

    Singleton._instances.pop(TicketOrder, None)
    Singleton._instances.pop(Config, None)

    yield

    Singleton._instances.pop(TicketOrder, None)
    Singleton._instances.pop(Config, None)


@pytest.mark.asyncio
async def test_oauth2_token_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token fetch should happen once until the cached token expires."""
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_ID", "client")
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_SECRET", "secret")
    monkeypatch.setenv("TICKETS_OAUTH2_TOKEN_URL", "https://idp.example.com/token")

    order = TicketOrder()
    fetch_mock = mock.AsyncMock(return_value=("cached-token", monotonic() + 120))
    monkeypatch.setattr(order, "_fetch_oauth2_token", fetch_mock)

    token_one = await order._get_oauth2_token()
    token_two = await order._get_oauth2_token()

    assert token_one == "cached-token"
    assert token_two == "cached-token"
    assert fetch_mock.await_count == 1


@pytest.mark.asyncio
async def test_ticket_validation_includes_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation API calls include Authorization when OAuth2 is enabled."""
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_ID", "client")
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_SECRET", "secret")
    monkeypatch.setenv("TICKETS_OAUTH2_TOKEN_URL", "https://idp.example.com/token")

    order = TicketOrder()
    monkeypatch.setattr(order, "_get_oauth2_token", mock.AsyncMock(return_value="tok-123"))

    fake_session = _FakeSession(response=_FakeResponse(status=HTTPStatus.NOT_FOUND))
    session_factory = mock.Mock(return_value=fake_session)
    monkeypatch.setattr(ticket_connector.aiohttp, "ClientSession", session_factory)

    data = await order.get_ticket_type(order="ABCD1", full_name="Jane Doe")

    assert data is None
    assert len(fake_session.post_calls) == 1
    headers = fake_session.post_calls[0]["kwargs"]["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer tok-123"


@pytest.mark.asyncio
async def test_ticket_validation_skips_call_when_token_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If OAuth2 is configured but no token is available, the API call is skipped."""
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_ID", "client")
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_SECRET", "secret")
    monkeypatch.setenv("TICKETS_OAUTH2_TOKEN_URL", "https://idp.example.com/token")

    order = TicketOrder()
    monkeypatch.setattr(order, "_get_oauth2_token", mock.AsyncMock(return_value=None))

    session_factory = mock.Mock()
    monkeypatch.setattr(ticket_connector.aiohttp, "ClientSession", session_factory)

    data = await order.get_ticket_type(order="ABCD1", full_name="Jane Doe")

    assert data is None
    session_factory.assert_not_called()


@pytest.mark.asyncio
async def test_ticket_refresh_includes_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh API calls include Authorization when OAuth2 is enabled."""
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_ID", "client")
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_SECRET", "secret")
    monkeypatch.setenv("TICKETS_OAUTH2_TOKEN_URL", "https://idp.example.com/token")

    order = TicketOrder()
    monkeypatch.setattr(order, "_get_oauth2_token", mock.AsyncMock(return_value="tok-123"))

    fake_session = _FakeSession(response=_FakeResponse(status=HTTPStatus.OK))
    monkeypatch.setattr(ticket_connector.aiohttp, "ClientSession", mock.Mock(return_value=fake_session))

    result = await order._update_tickets("https://val.example.com/tickets/refresh_all/")

    assert result is True
    assert len(fake_session.get_calls) == 1
    headers = fake_session.get_calls[0]["kwargs"]["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer tok-123"


@pytest.mark.asyncio
async def test_ticket_refresh_skips_call_when_token_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If OAuth2 is configured but no token is available, the refresh call is skipped."""
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_ID", "client")
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_SECRET", "secret")
    monkeypatch.setenv("TICKETS_OAUTH2_TOKEN_URL", "https://idp.example.com/token")

    order = TicketOrder()
    monkeypatch.setattr(order, "_get_oauth2_token", mock.AsyncMock(return_value=None))

    session_factory = mock.Mock()
    monkeypatch.setattr(ticket_connector.aiohttp, "ClientSession", session_factory)

    result = await order._update_tickets("https://val.example.com/tickets/refresh_all/")

    assert result is False
    session_factory.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_oauth2_token_rejects_non_dict_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON body that is not an object is rejected instead of raising AttributeError."""
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_ID", "client")
    monkeypatch.setenv("TICKETS_OAUTH2_CLIENT_SECRET", "secret")
    monkeypatch.setenv("TICKETS_OAUTH2_TOKEN_URL", "https://idp.example.com/token")

    order = TicketOrder()

    fake_session = _FakeSession(response=_FakeResponse(status=HTTPStatus.OK, json_payload=["unexpected"]))
    monkeypatch.setattr(ticket_connector.aiohttp, "ClientSession", mock.Mock(return_value=fake_session))

    result = await order._fetch_oauth2_token()

    assert result is None
