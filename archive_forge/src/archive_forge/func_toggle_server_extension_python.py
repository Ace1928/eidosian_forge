from __future__ import annotations
import logging
import os
import sys
import typing as t
from jupyter_core.application import JupyterApp
from jupyter_core.paths import ENV_CONFIG_PATH, SYSTEM_CONFIG_PATH, jupyter_config_dir
from tornado.log import LogFormatter
from traitlets import Bool
from jupyter_server._version import __version__
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager, ExtensionPackage
def toggle_server_extension_python(import_name: str, enabled: bool | None=None, parent: t.Any=None, user: bool=False, sys_prefix: bool=True) -> None:
    """Toggle the boolean setting for a given server extension
    in a Jupyter config file.
    """
    sys_prefix = False if user else sys_prefix
    config_dir = _get_config_dir(user=user, sys_prefix=sys_prefix)
    manager = ExtensionConfigManager(read_config_path=[config_dir], write_config_dir=os.path.join(config_dir, 'jupyter_server_config.d'))
    if enabled:
        manager.enable(import_name)
    else:
        manager.disable(import_name)