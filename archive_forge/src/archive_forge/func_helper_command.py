import copy
import enum
import functools
import logging
import multiprocessing
import shlex
import sys
import threading
from oslo_config import cfg
from oslo_config import types
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import daemon
def helper_command(self, sockpath):
    if self.pypath is None:
        raise AssertionError('helper_command requires priv_context pypath to be specified')
    if importutils.import_class(self.pypath) is not self:
        raise AssertionError('helper_command requires priv_context pypath for context object')
    if self.conf.helper_command:
        cmd = shlex.split(self.conf.helper_command)
    else:
        cmd = _HELPER_COMMAND_PREFIX + ['privsep-helper']
        try:
            for cfg_file in cfg.CONF.config_file:
                cmd.extend(['--config-file', cfg_file])
        except cfg.NoSuchOptError:
            pass
        try:
            if cfg.CONF.config_dir is not None:
                for cfg_dir in cfg.CONF.config_dir:
                    cmd.extend(['--config-dir', cfg_dir])
        except cfg.NoSuchOptError:
            pass
    cmd.extend(['--privsep_context', self.pypath, '--privsep_sock_path', sockpath])
    return cmd