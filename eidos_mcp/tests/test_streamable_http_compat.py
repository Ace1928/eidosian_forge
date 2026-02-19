import json

from starlette.testclient import TestClient

from eidos_mcp.eidos_mcp_server import _build_streamable_http_app, _merge_accept_header, mcp


def _initialize_payload() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "compat-test", "version": "1.0"},
        },
    }


def _fresh_app(
    enable_compat_headers: bool,
    *,
    enable_session_recovery: bool = True,
    enable_error_response_compat: bool = True,
    enforce_origin: bool = True,
):
    mcp._session_manager = None  # type: ignore[attr-defined]
    return _build_streamable_http_app(
        "/mcp",
        enable_compat_headers=enable_compat_headers,
        enable_session_recovery=enable_session_recovery,
        enable_error_response_compat=enable_error_response_compat,
        enforce_origin=enforce_origin,
    )


def test_merge_accept_header_appends_required_media_types() -> None:
    merged = _merge_accept_header("application/json", ("application/json", "text/event-stream"))
    assert "application/json" in merged
    assert "text/event-stream" in merged


def test_missing_content_type_is_upgraded_when_compat_enabled() -> None:
    app = _fresh_app(enable_compat_headers=True)
    payload = _initialize_payload()

    with TestClient(app, base_url="http://127.0.0.1:8928") as client:
        response = client.post("/mcp", content=json.dumps(payload))

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert "mcp-session-id" in response.headers


def test_mcp_origin_guard_rejects_untrusted_origin() -> None:
    app = _fresh_app(enable_compat_headers=True, enforce_origin=True)
    payload = _initialize_payload()

    with TestClient(app, base_url="http://127.0.0.1:8928") as client:
        response = client.post(
            "/mcp",
            content=json.dumps(payload),
            headers={"origin": "https://evil.example"},
        )

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "forbidden_origin"


def test_mcp_origin_guard_allows_localhost_origin() -> None:
    app = _fresh_app(enable_compat_headers=True, enforce_origin=True)
    payload = _initialize_payload()

    with TestClient(app, base_url="http://127.0.0.1:8928") as client:
        response = client.post(
            "/mcp",
            content=json.dumps(payload),
            headers={"origin": "http://localhost"},
        )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")


def test_non_json_content_type_is_upgraded_when_compat_enabled() -> None:
    app = _fresh_app(enable_compat_headers=True)
    payload = _initialize_payload()

    with TestClient(app, base_url="http://127.0.0.1:8928") as client:
        response = client.post(
            "/mcp",
            content=json.dumps(payload),
            headers={"Content-Type": "text/plain"},
        )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")


def test_missing_content_type_fails_without_compat_middleware() -> None:
    app = _fresh_app(enable_compat_headers=False)
    payload = _initialize_payload()

    with TestClient(app, base_url="http://127.0.0.1:8928") as client:
        response = client.post("/mcp", content=json.dumps(payload))

    assert response.status_code == 400


def test_stale_session_id_is_recovered_to_fresh_session() -> None:
    app = _fresh_app(
        enable_compat_headers=True,
        enable_session_recovery=True,
        enable_error_response_compat=True,
    )
    payload = _initialize_payload()

    with TestClient(app, base_url="http://127.0.0.1:8928") as client:
        response = client.post(
            "/mcp",
            content=json.dumps(payload),
            headers={"mcp-session-id": "stale-session-id"},
        )

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert response.headers.get("mcp-session-id")


def test_invalid_session_id_is_rewritten_to_json_error_without_recovery() -> None:
    app = _fresh_app(
        enable_compat_headers=True,
        enable_session_recovery=False,
        enable_error_response_compat=True,
    )
    payload = _initialize_payload()

    with TestClient(app, base_url="http://127.0.0.1:8928") as client:
        response = client.post(
            "/mcp",
            content=json.dumps(payload),
            headers={"mcp-session-id": "stale-session-id"},
        )

    assert response.status_code == 400
    assert response.headers.get("content-type", "").startswith("application/json")
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    assert body["error"]["message"] == "Bad Request: No valid session ID provided"


def test_invalid_session_id_without_recovery_or_response_compat_preserves_legacy_response() -> None:
    app = _fresh_app(
        enable_compat_headers=True,
        enable_session_recovery=False,
        enable_error_response_compat=False,
    )
    payload = _initialize_payload()

    with TestClient(app, base_url="http://127.0.0.1:8928") as client:
        response = client.post(
            "/mcp",
            content=json.dumps(payload),
            headers={"mcp-session-id": "stale-session-id"},
        )

    assert response.status_code == 400
    assert response.headers.get("content-type") is None
