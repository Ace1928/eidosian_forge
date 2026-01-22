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
@theme.setter
def theme(self, theme: Theme | str | None) -> None:
    theme = default_theme if theme is None else theme
    if isinstance(theme, str):
        try:
            theme = built_in_themes[theme]
        except KeyError:
            raise ValueError(f'{theme} is not a built-in theme; available themes are {nice_join(built_in_themes)}')
    if not isinstance(theme, Theme):
        raise ValueError('Theme must be a string or an instance of the Theme class')
    if self._theme is theme:
        return
    self._theme = theme
    for model in self.models:
        self._theme.apply_to_model(model)