from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
@property
def extension_apps(self):
    """Return mapping of extension names and sets of ExtensionApp objects."""
    return {name: {point.app for point in extension.extension_points.values() if point.app} for name, extension in self.extensions.items()}