from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
def from_jpserver_extensions(self, jpserver_extensions):
    """Add extensions from 'jpserver_extensions'-like dictionary."""
    for name, enabled in jpserver_extensions.items():
        self.add_extension(name, enabled=enabled)