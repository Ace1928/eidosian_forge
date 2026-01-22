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
def get_installed_packages_locale(locale_: str) -> tuple[dict, str]:
    """
    Get all jupyterlab extensions installed that contain locale data.

    Returns
    -------
    tuple
        A tuple in the form `(locale_data_dict, message)`,
        where the `locale_data_dict` is an ordered list
        of available language packs:
            >>> {"package-name": locale_data, ...}

    Examples
    --------
    - `entry_points={"jupyterlab.locale": "package-name = package_module"}`
    - `entry_points={"jupyterlab.locale": "jupyterlab-git = jupyterlab_git"}`
    """
    found_package_locales, message = _get_installed_package_locales()
    packages_locale_data = {}
    messages = message.split('\n')
    if not message:
        for package_name, package_root_path in found_package_locales.items():
            locales = {}
            try:
                locale_path = os.path.join(package_root_path, LOCALE_DIR)
                locales = {loc.lower(): loc for loc in os.listdir(locale_path) if os.path.isdir(os.path.join(locale_path, loc))}
            except Exception:
                messages.append(traceback.format_exc())
            if locale_.lower() in locales:
                locale_json_path = os.path.join(locale_path, locales[locale_.lower()], LC_MESSAGES_DIR, f'{package_name}.json')
                if os.path.isfile(locale_json_path):
                    try:
                        with open(locale_json_path, encoding='utf-8') as fh:
                            packages_locale_data[package_name] = json.load(fh)
                    except Exception:
                        messages.append(traceback.format_exc())
    return (packages_locale_data, '\n'.join(messages))