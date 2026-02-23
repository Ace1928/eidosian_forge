"""Fetch/list MCP resources via stdio or HTTP transports."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
except Exception:  # pragma: no cover - optional runtime dependency
    httpx = None

from eidosian_core import eidosian
from eidosian_core.ports import get_service_url
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import FORGE_ROOT

MCP_URL = get_service_url("eidos_mcp", default_port=8928, default_host="localhost", default_path="/mcp")


def _build_parser() -> argparse.ArgumentParser:
    default_python = os.environ.get("EIDOS_PYTHON_BIN", str(FORGE_ROOT / "eidosian_venv" / "bin" / "python3"))
    if not Path(default_python).exists():
        default_python = sys.executable

    parser = argparse.ArgumentParser(description="Fetch or list MCP resources.")
    parser.add_argument("uri", nargs="?", help="Resource URI (e.g. eidos://persona)")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available resources instead of reading a single resource.",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "stdio", "http"],
        default=os.environ.get("EIDOS_FETCH_TRANSPORT", "auto"),
        help="Transport mode (default: auto)",
    )
    parser.add_argument(
        "--python",
        default=default_python,
        help="Python executable to launch the MCP server in stdio mode.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("EIDOS_MCP_URL", MCP_URL),
        help="MCP server URL for HTTP mode.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON payload.")
    parser.add_argument(
        "--json-errors",
        action="store_true",
        help="Emit structured JSON on errors even in non-JSON mode.",
    )
    return parser


def _augment_pythonpath(env: dict[str, str]) -> None:
    pythonpath = env.get("PYTHONPATH", "")
    forge_dir = os.environ.get("EIDOS_FORGE_DIR", str(FORGE_ROOT))
    mcp_src = os.path.join(forge_dir, "eidos_mcp", "src")
    lib_path = os.path.join(forge_dir, "lib")
    paths_to_add = [mcp_src, forge_dir, lib_path]
    for path in paths_to_add:
        if path not in pythonpath:
            pythonpath = f"{path}:{pythonpath}" if pythonpath else path
    env["PYTHONPATH"] = pythonpath


def _serialize_blob(value: Any) -> tuple[str, str]:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii"), "base64"
    return str(value), "plain"


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_content_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        data = item.model_dump()
    elif isinstance(item, dict):
        data = dict(item)
    else:
        data = {
            "uri": getattr(item, "uri", None),
            "mimeType": getattr(item, "mimeType", None),
            "text": getattr(item, "text", None),
            "blob": getattr(item, "blob", None),
            "type": item.__class__.__name__,
        }

    normalized: dict[str, Any] = {
        "uri": _json_scalar(data.get("uri")),
        "mimeType": _json_scalar(data.get("mimeType") or data.get("mime_type")),
        "type": _json_scalar(data.get("type") or item.__class__.__name__),
    }

    text = data.get("text")
    blob = data.get("blob")
    if text is not None:
        normalized["kind"] = "text"
        normalized["text"] = str(text)
    elif blob is not None:
        encoded, encoding = _serialize_blob(blob)
        normalized["kind"] = "blob"
        normalized["blob"] = encoded
        normalized["blob_encoding"] = encoding
    else:
        normalized["kind"] = "unknown"

    return normalized


def _normalize_resource_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        data = item.model_dump()
    elif isinstance(item, dict):
        data = dict(item)
    else:
        data = {
            "uri": getattr(item, "uri", None),
            "name": getattr(item, "name", None),
            "description": getattr(item, "description", None),
            "mimeType": getattr(item, "mimeType", None),
        }

    return {
        "uri": _json_scalar(data.get("uri")),
        "name": _json_scalar(data.get("name")),
        "description": _json_scalar(data.get("description")),
        "mimeType": _json_scalar(data.get("mimeType") or data.get("mime_type")),
    }


def _extract_text(contents: list[dict[str, Any]]) -> Optional[str]:
    for item in contents:
        text = item.get("text")
        if isinstance(text, str):
            return text
    return None


def _stdio_params(python_bin: str) -> StdioServerParameters:
    server_args = ["-m", "eidos_mcp.eidos_mcp_server"]
    env = dict(os.environ)
    _augment_pythonpath(env)
    env["EIDOS_MCP_TRANSPORT"] = "stdio"
    env.pop("EIDOS_MCP_MOUNT_PATH", None)
    env.setdefault("EIDOS_GIS_PATH", str(Path.home() / ".eidosian" / "gis_data.local.json"))
    return StdioServerParameters(command=python_bin, args=server_args, env=env)


async def _fetch_stdio(uri: str, python_bin: str) -> dict[str, Any]:
    params = _stdio_params(python_bin)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource(uri)
            contents = [_normalize_content_item(item) for item in (result.contents or [])]
            return {
                "success": True,
                "transport": "stdio",
                "uri": uri,
                "content_count": len(contents),
                "contents": contents,
            }


async def _list_resources_stdio(python_bin: str) -> dict[str, Any]:
    params = _stdio_params(python_bin)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_resources()
            resources = [_normalize_resource_item(item) for item in (result.resources or [])]
            return {
                "success": True,
                "transport": "stdio",
                "count": len(resources),
                "resources": resources,
            }


async def _fetch_http(uri: str, url: str) -> dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx is not installed for HTTP transport")

    async with httpx.AsyncClient(timeout=8.0) as client:
        init_resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "eidos_fetch", "version": "1.0.0"},
                },
            },
            headers={"Content-Type": "application/json"},
        )
        init_resp.raise_for_status()

        read_resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "method": "resources/read",
                "id": 2,
                "params": {"uri": uri},
            },
            headers={"Content-Type": "application/json"},
        )
        read_resp.raise_for_status()
        data = read_resp.json()
        result = data.get("result", {})
        contents = [_normalize_content_item(item) for item in (result.get("contents") or [])]
        return {
            "success": True,
            "transport": "http",
            "uri": uri,
            "content_count": len(contents),
            "contents": contents,
        }


async def _list_resources_http(url: str) -> dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx is not installed for HTTP transport")

    async with httpx.AsyncClient(timeout=8.0) as client:
        init_resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": 1,
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "eidos_fetch", "version": "1.0.0"},
                },
            },
            headers={"Content-Type": "application/json"},
        )
        init_resp.raise_for_status()

        list_resp = await client.post(
            url,
            json={
                "jsonrpc": "2.0",
                "method": "resources/list",
                "id": 2,
                "params": {},
            },
            headers={"Content-Type": "application/json"},
        )
        list_resp.raise_for_status()
        data = list_resp.json()
        result = data.get("result", {})
        resources = [_normalize_resource_item(item) for item in (result.get("resources") or [])]
        return {
            "success": True,
            "transport": "http",
            "count": len(resources),
            "resources": resources,
        }


def _run_with_transport(args: argparse.Namespace) -> dict[str, Any]:
    def run_fetch(transport: str) -> dict[str, Any]:
        if args.list:
            if transport == "stdio":
                return asyncio.run(_list_resources_stdio(args.python))
            return asyncio.run(_list_resources_http(args.url))
        if transport == "stdio":
            return asyncio.run(_fetch_stdio(str(args.uri), args.python))
        return asyncio.run(_fetch_http(str(args.uri), args.url))

    transport = str(args.transport or "auto").strip().lower()
    if transport in {"stdio", "http"}:
        return run_fetch(transport)

    stdio_error: Optional[Exception] = None
    try:
        return run_fetch("stdio")
    except Exception as exc:
        stdio_error = exc

    try:
        return run_fetch("http")
    except Exception as exc:
        if stdio_error is None:
            raise
        raise RuntimeError(f"auto transport failed: stdio={stdio_error}; http={exc}") from exc


@eidosian()
def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.list and not args.uri:
        parser.error("uri is required unless --list is provided")

    try:
        payload = _run_with_transport(args)
    except Exception as exc:
        err = {
            "success": False,
            "error": str(exc),
            "transport": args.transport,
            "mode": "list" if args.list else "read",
            "uri": args.uri,
        }
        if args.json or args.json_errors:
            print(json.dumps(err, indent=2))
        else:
            print(f"eidos_fetch error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if args.list:
        for resource in payload.get("resources", []):
            uri = resource.get("uri") or "<unknown-uri>"
            name = resource.get("name")
            if name:
                print(f"{uri}\t{name}")
            else:
                print(uri)
        return 0

    contents = payload.get("contents") or []
    text = _extract_text(contents)
    if text is not None:
        print(text)
        return 0

    # Non-text resources are emitted as JSON summaries for safe downstream handling.
    print(
        json.dumps(
            {
                "uri": payload.get("uri"),
                "transport": payload.get("transport"),
                "content_count": payload.get("content_count", len(contents)),
                "contents": contents,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
