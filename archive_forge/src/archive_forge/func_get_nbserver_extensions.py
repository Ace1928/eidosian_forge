import os
import types
import inspect
from functools import wraps
from jupyter_core.paths import jupyter_config_path
from traitlets.traitlets import is_trait
from jupyter_server.services.config.manager import ConfigManager
from .traits import NotebookAppTraits
def get_nbserver_extensions(config_dirs):
    cm = ConfigManager(read_config_path=config_dirs)
    section = cm.get('jupyter_notebook_config')
    extensions = section.get('NotebookApp', {}).get('nbserver_extensions', {})
    return extensions