import contextlib
import errno
import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import FrozenSet, Optional
from urllib.error import URLError
from urllib.request import Request, quote, urljoin, urlopen
from jupyter_core.paths import jupyter_config_dir
from jupyter_server.extension.serverextension import GREEN_ENABLED, GREEN_OK, RED_DISABLED, RED_X
from jupyterlab_server.config import (
from jupyterlab_server.process import Process, WatchHelper, list2cmdline, which
from packaging.version import Version
from traitlets import Bool, HasTraits, Instance, List, Unicode, default
from jupyterlab._version import __version__
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.jlpmapp import HERE, YARN_PATH
from jupyterlab.semver import Range, gt, gte, lt, lte, make_semver
def toggle_extension(self, extension, value, level='sys_prefix'):
    """Enable or disable a lab extension.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
    app_settings_dir = osp.join(self.app_dir, 'settings')
    page_config = get_static_page_config(app_settings_dir=app_settings_dir, logger=self.logger, level=level)
    disabled = page_config.get('disabledExtensions', {})
    did_something = False
    is_disabled = disabled.get(extension, False)
    if value and (not is_disabled):
        disabled[extension] = True
        did_something = True
    elif not value and is_disabled:
        disabled[extension] = False
        did_something = True
    if did_something:
        page_config['disabledExtensions'] = disabled
        write_page_config(page_config, level=level)
    return did_something