#!/usr/bin/env python3
"""Apply the Eidos parity sync profile to a local Syncthing instance.

This script is intentionally portable so it can be run on both laptop and
desktop with opposite remote-device settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROFILE: list[dict[str, str]] = [
    {"id": "dot-eidosian", "label": ".eidosian", "path": "{home}/.eidosian"},
    {"id": "dot-ruff-cache", "label": ".ruff_cache", "path": "{home}/.ruff_cache"},
    {"id": "dot-benchmarks", "label": ".benchmarks", "path": "{home}/.benchmarks"},
    {"id": "dot-gnupg", "label": ".gnupg", "path": "{home}/.gnupg"},
    {"id": "dot-config", "label": ".config", "path": "{home}/.config"},
    {"id": "dot-codex", "label": ".codex", "path": "{home}/.codex"},
    {"id": "dot-local-share-backgrounds", "label": ".local/share/backgrounds", "path": "{home}/.local/share/backgrounds"},
    {"id": "dot-local-share-wallpapers", "label": ".local/share/wallpapers", "path": "{home}/.local/share/wallpapers"},
    {"id": "dot-local-share-plasma", "label": ".local/share/plasma", "path": "{home}/.local/share/plasma"},
    {"id": "dot-local-share-icons", "label": ".local/share/icons", "path": "{home}/.local/share/icons"},
    {"id": "dot-gemini", "label": ".gemini", "path": "{home}/.gemini"},
    {"id": "dot-copilot", "label": ".copilot", "path": "{home}/.copilot"},
    {"id": "dot-continue", "label": ".continue", "path": "{home}/.continue"},
    {"id": "dot-cursor", "label": ".cursor", "path": "{home}/.cursor"},
    {"id": "dot-vscode", "label": ".vscode", "path": "{home}/.vscode"},
    {"id": "dot-chatgpt-local", "label": ".chatgpt-local", "path": "{home}/.chatgpt-local"},
    {"id": "dot-chatmock", "label": ".chatmock", "path": "{home}/.chatmock"},
    {"id": "dot-eidos-core", "label": ".eidos_core", "path": "{home}/.eidos_core"},
    {"id": "dot-eidos", "label": ".eidos", "path": "{home}/.eidos"},
    {"id": "dot-moltbot", "label": ".moltbot", "path": "{home}/.moltbot"},
    {"id": "dot-local-bin", "label": ".local/bin", "path": "{home}/.local/bin"},
    {"id": "dot-local-share-konsole", "label": ".local/share/konsole", "path": "{home}/.local/share/konsole"},
    {"id": "dot-local-share-kscreen", "label": ".local/share/kscreen", "path": "{home}/.local/share/kscreen"},
    {"id": "dot-local-share-kwin", "label": ".local/share/kwin", "path": "{home}/.local/share/kwin"},
    {"id": "dot-local-share-kded5", "label": ".local/share/kded5", "path": "{home}/.local/share/kded5"},
    {"id": "dot-local-share-klipper", "label": ".local/share/klipper", "path": "{home}/.local/share/klipper"},
    {
        "id": "dot-local-share-applications",
        "label": ".local/share/applications",
        "path": "{home}/.local/share/applications",
    },
    {
        "id": "dot-local-share-plasma-systemmonitor",
        "label": ".local/share/plasma-systemmonitor",
        "path": "{home}/.local/share/plasma-systemmonitor",
    },
    {"id": "home-pictures", "label": "Pictures", "path": "{home}/Pictures"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-id", required=True, help="Remote Syncthing device ID.")
    parser.add_argument("--remote-name", default="Eidos-Peer", help="Remote device display name.")
    parser.add_argument(
        "--remote-addr",
        action="append",
        default=[],
        help="Remote device address (repeatable), e.g. tcp://100.127.151.46:22000",
    )
    parser.add_argument(
        "--home",
        default=str(Path.home()),
        help="Home directory whose folders should be mapped (default: current user home).",
    )
    parser.add_argument(
        "--syncthing-config",
        default="",
        help="Path to syncthing config.xml. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8384",
        help="Syncthing API base URL (default: http://127.0.0.1:8384).",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Do not restart syncthing after config update.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def detect_config_path(explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            fail(f"Syncthing config path does not exist: {path}")
        return path

    candidates = [
        Path.home() / ".local" / "state" / "syncthing" / "config.xml",
        Path.home() / ".config" / "syncthing" / "config.xml",
    ]
    for path in candidates:
        if path.is_file():
            return path

    fail("Could not auto-detect syncthing config.xml.")
    return Path()  # Unreachable.


def read_api_key(config_xml_path: Path) -> str:
    content = config_xml_path.read_text(encoding="utf-8")
    match = re.search(r"<apikey>([^<]+)</apikey>", content)
    if not match:
        fail(f"Could not find <apikey> in {config_xml_path}")
    return match.group(1)


def api_call(base_url: str, api_key: str, method: str, endpoint: str, payload: Any | None = None) -> Any:
    url = f"{base_url.rstrip('/')}{endpoint}"
    data = None
    headers = {"X-API-Key": api_key}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_remote_device(config: dict[str, Any], remote_id: str, remote_name: str, remote_addrs: list[str]) -> bool:
    changed = False
    devices = config.setdefault("devices", [])
    defaults = copy.deepcopy(config.get("defaults", {}).get("device", {}))

    merged_addrs = ["dynamic", *remote_addrs]
    unique_addrs: list[str] = []
    for address in merged_addrs:
        if address and address not in unique_addrs:
            unique_addrs.append(address)

    desired = {
        **defaults,
        "deviceID": remote_id,
        "name": remote_name,
        "addresses": unique_addrs,
        "compression": "metadata",
        "introducer": False,
        "autoAcceptFolders": True,
        "paused": False,
    }

    existing = next((device for device in devices if device.get("deviceID") == remote_id), None)
    if existing is None:
        devices.append(desired)
        return True

    for key, value in desired.items():
        if existing.get(key) != value:
            existing[key] = value
            changed = True

    return changed


def folder_devices(local_id: str, remote_id: str) -> list[dict[str, str]]:
    return [
        {"deviceID": local_id, "introducedBy": "", "encryptionPassword": ""},
        {"deviceID": remote_id, "introducedBy": "", "encryptionPassword": ""},
    ]


def ensure_folder(
    config: dict[str, Any],
    spec: dict[str, str],
    home_dir: Path,
    local_id: str,
    remote_id: str,
) -> bool:
    changed = False
    folders = config.setdefault("folders", [])
    defaults = copy.deepcopy(config.get("defaults", {}).get("folder", {}))

    folder_path = Path(spec["path"].format(home=str(home_dir)))
    ensure_dir(folder_path)

    desired = {
        **defaults,
        "id": spec["id"],
        "label": spec["label"],
        "path": str(folder_path),
        "type": "sendreceive",
        "rescanIntervalS": 120,
        "fsWatcherEnabled": True,
        "fsWatcherDelayS": 5,
        "ignorePerms": False,
        "ignoreDelete": False,
        "autoNormalize": True,
        "paused": False,
        "devices": folder_devices(local_id=local_id, remote_id=remote_id),
    }

    existing = next((folder for folder in folders if folder.get("id") == spec["id"]), None)
    if existing is None:
        folders.append(desired)
        return True

    keys_to_update = [
        "label",
        "path",
        "type",
        "rescanIntervalS",
        "fsWatcherEnabled",
        "fsWatcherDelayS",
        "ignorePerms",
        "ignoreDelete",
        "autoNormalize",
        "paused",
        "devices",
    ]
    for key in keys_to_update:
        value = desired[key]
        if existing.get(key) != value:
            existing[key] = value
            changed = True

    return changed


def main() -> None:
    args = parse_args()
    home_dir = Path(args.home).expanduser().resolve()
    config_xml = detect_config_path(args.syncthing_config)
    api_key = read_api_key(config_xml)

    status = api_call(args.api_url, api_key, "GET", "/rest/system/status")
    local_id = status.get("myID")
    if not local_id:
        fail("Could not determine local device ID from Syncthing status.")

    config = api_call(args.api_url, api_key, "GET", "/rest/config")
    original = json.dumps(config, sort_keys=True)

    changed = False
    changed |= ensure_remote_device(
        config=config,
        remote_id=args.remote_id,
        remote_name=args.remote_name,
        remote_addrs=args.remote_addr,
    )

    for spec in PROFILE:
        changed |= ensure_folder(
            config=config,
            spec=spec,
            home_dir=home_dir,
            local_id=local_id,
            remote_id=args.remote_id,
        )

    updated = json.dumps(config, sort_keys=True)
    if changed and updated != original:
        api_call(args.api_url, api_key, "PUT", "/rest/config", payload=config)
        print("Applied parity profile config update.")
        if not args.no_restart:
            api_call(args.api_url, api_key, "POST", "/rest/system/restart")
            print("Requested Syncthing restart.")
            time.sleep(5)
            api_key = read_api_key(config_xml)
    else:
        print("Parity profile already up to date.")

    for spec in PROFILE:
        folder_id = urllib.parse.quote(spec["id"], safe="")
        endpoint = f"/rest/db/scan?folder={folder_id}"
        api_call(args.api_url, api_key, "POST", endpoint)

    print("Triggered scans for all parity profile folders.")
    print(f"Profile folder count: {len(PROFILE)}")


if __name__ == "__main__":
    main()
