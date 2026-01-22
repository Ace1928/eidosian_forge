import datetime
import functools
import hashlib
import json
import logging
import optparse
import os.path
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.rich.console import Group
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import get_default_environment
from pip._internal.metadata.base import DistributionVersion
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.entrypoints import (
from pip._internal.utils.filesystem import adjacent_tmp_file, check_path_owner, replace
from pip._internal.utils.misc import ensure_dir
def pip_self_version_check(session: PipSession, options: optparse.Values) -> None:
    """Check for an update for pip.

    Limit the frequency of checks to once per week. State is stored either in
    the active virtualenv or in the user's USER_CACHE_DIR keyed off the prefix
    of the pip script path.
    """
    installed_dist = get_default_environment().get_distribution('pip')
    if not installed_dist:
        return
    try:
        upgrade_prompt = _self_version_check_logic(state=SelfCheckState(cache_dir=options.cache_dir), current_time=datetime.datetime.now(datetime.timezone.utc), local_version=installed_dist.version, get_remote_version=functools.partial(_get_current_remote_pip_version, session, options))
        if upgrade_prompt is not None:
            logger.warning('%s', upgrade_prompt, extra={'rich': True})
    except Exception:
        logger.warning('There was an error checking the latest version of pip.')
        logger.debug('See below for error', exc_info=True)