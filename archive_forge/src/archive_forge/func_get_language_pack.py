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
def get_language_pack(locale_: str) -> tuple:
    """
    Get a language pack for a given `locale_` and update with any installed
    package locales.

    Returns
    -------
    tuple
        A tuple in the form `(locale_data_dict, message)`.

    Notes
    -----
    We call `_get_installed_language_pack_locales` via a subprocess to
    guarantee the results represent the most up-to-date entry point
    information, which seems to be defined on interpreter startup.
    """
    found_locales, message = _get_installed_language_pack_locales()
    found_packages_locales, message = get_installed_packages_locale(locale_)
    locale_data = {}
    messages = message.split('\n')
    if not message and is_valid_locale(locale_) and (locale_ in found_locales):
        path = found_locales[locale_]
        for root, __, files in os.walk(path, topdown=False):
            for name in files:
                if name.endswith('.json'):
                    pkg_name = name.replace('.json', '')
                    json_path = os.path.join(root, name)
                    try:
                        with open(json_path, encoding='utf-8') as fh:
                            merged_data = json.load(fh)
                    except Exception:
                        messages.append(traceback.format_exc())
                    if pkg_name in found_packages_locales:
                        pkg_data = found_packages_locales[pkg_name]
                        merged_data = merge_locale_data(merged_data, pkg_data)
                    locale_data[pkg_name] = merged_data
        for pkg_name, data in found_packages_locales.items():
            if pkg_name not in locale_data:
                locale_data[pkg_name] = data
    return (locale_data, '\n'.join(messages))