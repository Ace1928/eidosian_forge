import logging
import os
import shutil
import subprocess
import sysconfig
import typing
import urllib.parse
from abc import ABC, abstractmethod
from functools import lru_cache
from os.path import commonprefix
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from pip._vendor.requests.auth import AuthBase, HTTPBasicAuth
from pip._vendor.requests.models import Request, Response
from pip._vendor.requests.utils import get_netrc_auth
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
from pip._internal.vcs.versioncontrol import AuthInfo
@lru_cache(maxsize=None)
def get_keyring_provider(provider: str) -> KeyRingBaseProvider:
    logger.verbose('Keyring provider requested: %s', provider)
    if KEYRING_DISABLED:
        provider = 'disabled'
    if provider in ['import', 'auto']:
        try:
            impl = KeyRingPythonProvider()
            logger.verbose('Keyring provider set: import')
            return impl
        except ImportError:
            pass
        except Exception as exc:
            msg = 'Installed copy of keyring fails with exception %s'
            if provider == 'auto':
                msg = msg + ', trying to find a keyring executable as a fallback'
            logger.warning(msg, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
    if provider in ['subprocess', 'auto']:
        cli = shutil.which('keyring')
        if cli and cli.startswith(sysconfig.get_path('scripts')):

            @typing.no_type_check
            def PATH_as_shutil_which_determines_it() -> str:
                path = os.environ.get('PATH', None)
                if path is None:
                    try:
                        path = os.confstr('CS_PATH')
                    except (AttributeError, ValueError):
                        path = os.defpath
                return path
            scripts = Path(sysconfig.get_path('scripts'))
            paths = []
            for path in PATH_as_shutil_which_determines_it().split(os.pathsep):
                p = Path(path)
                try:
                    if not p.samefile(scripts):
                        paths.append(path)
                except FileNotFoundError:
                    pass
            path = os.pathsep.join(paths)
            cli = shutil.which('keyring', path=path)
        if cli:
            logger.verbose('Keyring provider set: subprocess with executable %s', cli)
            return KeyRingCliProvider(cli)
    logger.verbose('Keyring provider set: disabled')
    return KeyRingNullProvider()