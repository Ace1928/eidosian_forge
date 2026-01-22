from __future__ import annotations
import logging
import pkgutil
import re
from collections.abc import Sequence
from typing import cast
import requests
from requests_file import FileAdapter  # type: ignore[import-untyped]
from .cache import DiskCache
def extract_tlds_from_suffix_list(suffix_list_text: str) -> tuple[list[str], list[str]]:
    """Parse the raw suffix list text for its different designations of suffixes."""
    public_text, _, private_text = suffix_list_text.partition(PUBLIC_PRIVATE_SUFFIX_SEPARATOR)
    public_tlds = [m.group('suffix') for m in PUBLIC_SUFFIX_RE.finditer(public_text)]
    private_tlds = [m.group('suffix') for m in PUBLIC_SUFFIX_RE.finditer(private_text)]
    return (public_tlds, private_tlds)