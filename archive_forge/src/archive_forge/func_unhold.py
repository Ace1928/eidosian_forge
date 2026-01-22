from __future__ import annotations
import logging # isort:skip
import gc
import weakref
from json import loads
from typing import TYPE_CHECKING, Any, Iterable
from jinja2 import Template
from ..core.enums import HoldPolicyType
from ..core.has_props import is_DataModel
from ..core.query import find, is_single_string_selector
from ..core.serialization import (
from ..core.templates import FILE
from ..core.types import ID
from ..core.validation import check_integrity, process_validation_issues
from ..events import Event
from ..model import Model
from ..themes import Theme, built_in_themes, default as default_theme
from ..util.serialization import make_id
from ..util.strings import nice_join
from ..util.version import __version__
from .callbacks import (
from .events import (
from .json import DocJson, PatchJson
from .models import DocumentModelManager
from .modules import DocumentModuleManager
def unhold(self) -> None:
    """ Turn off any active document hold and apply any collected events.

        Returns:
            None

        """
    self.callbacks.unhold()