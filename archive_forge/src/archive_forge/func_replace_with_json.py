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
def replace_with_json(self, json: DocJson) -> None:
    """ Overwrite everything in this document with the JSON-encoded
        document.

        json (JSON-data) :
            A JSON-encoded document to overwrite this one.

        Returns:
            None

        """
    replacement = self.from_json(json)
    replacement._destructively_move(self)