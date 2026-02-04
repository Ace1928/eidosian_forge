#!/usr/bin/env python3
"""Export filtered Firefox cookies to Netscape cookie file for yt-dlp."""
from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Iterable, Optional


DEFAULT_DOMAINS = (
    "youtube.com",
    "google.com",
    "accounts.google.com",
    "googlevideo.com",
)


def _find_profiles_ini() -> Optional[Path]:
    candidates = [
        Path.home() / "snap" / "firefox" / "common" / ".mozilla" / "firefox" / "profiles.ini",
        Path.home() / ".mozilla" / "firefox" / "profiles.ini",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_default_profile(profiles_ini: Path) -> Optional[Path]:
    config = configparser.ConfigParser()
    config.read(profiles_ini)
    root = profiles_ini.parent
    for section in config.sections():
        if not section.startswith("Profile"):
            continue
        if config.get(section, "Default", fallback="0") != "1":
            continue
        path_value = config.get(section, "Path", fallback=None)
        if not path_value:
            continue
        is_relative = config.get(section, "IsRelative", fallback="1") == "1"
        return (root / path_value) if is_relative else Path(path_value)
    return None


def _pick_profile(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    profiles_ini = _find_profiles_ini()
    if profiles_ini is None:
        raise SystemExit("No Firefox profiles.ini found")
    profile = _load_default_profile(profiles_ini)
    if profile is None:
        raise SystemExit("No default Firefox profile found")
    return profile


def _copy_sqlite(src: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="gf_cookies_"))
    dest = temp_dir / "cookies.sqlite"
    shutil.copy2(src, dest)
    return dest


def _netscape_line(
    host: str,
    path: str,
    secure: bool,
    expiry: int,
    name: str,
    value: str,
) -> str:
    flag = "TRUE" if host.startswith(".") else "FALSE"
    secure_str = "TRUE" if secure else "FALSE"
    return "\t".join([host, flag, path, secure_str, str(expiry), name, value])


def export_cookies(
    profile: Path,
    output: Path,
    domains: Iterable[str],
    names: Optional[Iterable[str]] = None,
) -> int:
    cookies_db = profile / "cookies.sqlite"
    if not cookies_db.exists():
        raise SystemExit(f"cookies.sqlite not found in {profile}")
    temp_db = _copy_sqlite(cookies_db)
    conn = sqlite3.connect(temp_db)
    try:
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT host, path, isSecure, expiry, name, value FROM moz_cookies"
        ).fetchall()
    finally:
        conn.close()
    domain_set = tuple(domains)
    name_set = {name.strip() for name in (names or []) if name.strip()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# Netscape HTTP Cookie File\n")
        for host, path, is_secure, expiry, name, value in rows:
            host_str = str(host or "")
            if not any(dom in host_str for dom in domain_set):
                continue
            if name_set and name not in name_set:
                continue
            line = _netscape_line(
                host=host_str,
                path=str(path or "/"),
                secure=bool(is_secure),
                expiry=int(expiry or 0),
                name=str(name or ""),
                value=str(value or ""),
            )
            handle.write(line + "\n")
    return output.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Firefox cookies for yt-dlp.")
    parser.add_argument("--profile", help="Firefox profile directory")
    parser.add_argument(
        "--output",
        default=str(Path.cwd() / "glyph_forge_output" / "yt_cookies.txt"),
        help="Output cookie file path",
    )
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated list of domains to include",
    )
    parser.add_argument(
        "--names",
        default="",
        help="Comma-separated list of cookie names to include (optional)",
    )
    args = parser.parse_args()

    profile = _pick_profile(args.profile)
    output = Path(args.output).expanduser()
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    names = [item.strip() for item in args.names.split(",") if item.strip()]
    size = export_cookies(profile, output, domains, names=names or None)
    print(f"Exported cookies to {output} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
