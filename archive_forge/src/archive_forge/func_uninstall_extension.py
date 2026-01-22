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
def uninstall_extension(self, name):
    """Uninstall an extension by name.

        Returns `True` if a rebuild is recommended, `False` otherwise.
        """
    info = self.info
    logger = self.logger
    if name in info['federated_extensions']:
        if info['federated_extensions'][name].get('install', {}).get('uninstallInstructions', None):
            logger.error('JupyterLab cannot uninstall this extension. %s' % info['federated_extensions'][name]['install']['uninstallInstructions'])
        else:
            logger.error('JupyterLab cannot uninstall %s since it was installed outside of JupyterLab. Use the same method used to install this extension to uninstall this extension.' % name)
        return False
    if name in info['core_extensions']:
        config = self._read_build_config()
        uninstalled = config.get('uninstalled_core_extensions', [])
        if name not in uninstalled:
            logger.info('Uninstalling core extension %s' % name)
            uninstalled.append(name)
            config['uninstalled_core_extensions'] = uninstalled
            self._write_build_config(config)
            return True
        return False
    local = info['local_extensions']
    for extname, data in info['extensions'].items():
        path = data['path']
        if extname == name:
            msg = f'Uninstalling {name} from {osp.dirname(path)}'
            logger.info(msg)
            os.remove(path)
            if extname in local:
                config = self._read_build_config()
                data = config.setdefault('local_extensions', {})
                del data[extname]
                self._write_build_config(config)
            return True
    logger.warning('No labextension named "%s" installed' % name)
    return False