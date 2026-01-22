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
def latest_compatible_package_versions(self, names):
    """Get the latest compatible versions of several packages

        Like _latest_compatible_package_version, but optimized for
        retrieving the latest version for several packages in one go.
        """
    core_data = self.info['core_data']
    keys = []
    for name in names:
        try:
            metadata = _fetch_package_metadata(self.registry, name, self.logger)
        except URLError:
            continue
        versions = metadata.get('versions', {})

        def sort_key(key_value):
            return _semver_key(key_value[0], prerelease_first=True)
        for version, data in sorted(versions.items(), key=sort_key, reverse=True):
            if 'deprecated' in data:
                continue
            deps = data.get('dependencies', {})
            errors = _validate_compatibility(name, deps, core_data)
            if not errors:
                keys.append(f'{name}@{version}')
                break
    versions = {}
    if not keys:
        return versions
    with TemporaryDirectory() as tempdir:
        ret = self._run([which('npm'), 'pack', *keys], cwd=tempdir)
        if ret != 0:
            msg = '"%s" is not a valid npm package'
            raise ValueError(msg % keys)
        for key in keys:
            fname = key[0].replace('@', '') + key[1:].replace('@', '-').replace('/', '-') + '.tgz'
            data = read_package(osp.join(tempdir, fname))
            if not _validate_extension(data):
                versions[data['name']] = data['version']
    return versions