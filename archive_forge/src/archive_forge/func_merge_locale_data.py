from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
def merge_locale_data(language_pack_locale_data: dict[str, Any], package_locale_data: dict[str, Any]) -> dict[str, Any]:
    """
    Merge language pack data with locale data bundled in packages.

    Parameters
    ----------
    language_pack_locale_data: dict
        The dictionary with language pack locale data.
    package_locale_data: dict
        The dictionary with package locale data.

    Returns
    -------
    dict
        Merged locale data.
    """
    result = language_pack_locale_data
    package_lp_metadata = language_pack_locale_data.get('', {})
    package_lp_version = package_lp_metadata.get('version', None)
    package_lp_domain = package_lp_metadata.get('domain', None)
    package_metadata = package_locale_data.get('', {})
    package_version = package_metadata.get('version', None)
    package_domain = package_metadata.get('domain', 'None')
    if package_lp_version and package_version and (package_domain == package_lp_domain):
        package_version = parse_version(package_version)
        package_lp_version = parse_version(package_lp_version)
        if package_version > package_lp_version:
            result = language_pack_locale_data.copy()
            result.update(package_locale_data)
    return result