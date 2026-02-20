#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra in (FORGE_ROOT / "lib", FORGE_ROOT):
    extra_str = str(extra)
    if extra.exists() and extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from eidosian_core.ports import get_service_url

MCP_URL = get_service_url("eidos_mcp", default_port=8928, default_path="/mcp")
CODEX_SECTION = "[mcp_servers.eidosian_nexus]"
CODEX_URL_LINE = f'url = "{MCP_URL}"'


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{stamp}")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path


def configure_codex(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = 'model = "gpt-5.3-codex"\nmodel_reasoning_effort = "high"\n'

    if CODEX_SECTION in text:
        section_pattern = re.compile(
            r"(\[mcp_servers\.eidosian_nexus\][^\[]*)",
            re.MULTILINE | re.DOTALL,
        )
        match = section_pattern.search(text)
        if match:
            section = match.group(1)
            if re.search(r'^\s*url\s*=\s*".*"\s*$', section, re.MULTILINE):
                section = re.sub(
                    r'^\s*url\s*=\s*".*"\s*$',
                    CODEX_URL_LINE,
                    section,
                    flags=re.MULTILINE,
                )
            else:
                section = section.rstrip() + "\n" + CODEX_URL_LINE + "\n"
            text = text[: match.start(1)] + section + text[match.end(1) :]
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += f"\n{CODEX_SECTION}\n{CODEX_URL_LINE}\n"

    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def configure_gemini(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup(path)

    data: dict
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    mcp_servers = data.setdefault("mcpServers", {})
    mcp_servers["eidosian_nexus"] = {
        "httpUrl": MCP_URL,
        "url": MCP_URL,
        "trust": True,
    }

    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Idempotently configure Codex/Gemini MCP client settings.")
    parser.add_argument("--home", default=str(Path.home()), help="Home directory (default: current user home).")
    args = parser.parse_args()

    home = Path(args.home).expanduser().resolve()
    codex_path = home / ".codex" / "config.toml"
    gemini_path = home / ".gemini" / "settings.json"

    configure_codex(codex_path)
    configure_gemini(gemini_path)

    print(f"Configured Codex MCP: {codex_path}")
    print(f"Configured Gemini MCP: {gemini_path}")
    print(f"MCP URL: {MCP_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
