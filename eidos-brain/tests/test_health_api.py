from api import create_app
from core.health import HealthChecker


def run_app(path: str) -> tuple[str, bytes]:
    """Utility to run the WSGI app for ``path`` and capture response."""

    app = create_app(HealthChecker())
    captured: dict[str, str] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = dict(headers)

    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": path,
        "SERVER_NAME": "test",
        "SERVER_PORT": "80",
        "wsgi.input": b"",
    }

    result = b"".join(app(environ, start_response))
    return captured.get("status", ""), result


def test_healthz_returns_ok() -> None:
    status, body = run_app("/healthz")
    assert status.startswith("200")
    assert body == b'{"status": "ok"}'


def test_unknown_path_returns_404() -> None:
    status, _ = run_app("/missing")
    assert status.startswith("404")
