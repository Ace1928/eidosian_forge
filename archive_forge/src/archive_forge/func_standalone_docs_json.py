from __future__ import annotations
import logging # isort:skip
import re
from contextlib import contextmanager
from typing import (
from weakref import WeakKeyDictionary
from ..core.types import ID
from ..document.document import Document
from ..model import Model, collect_models
from ..settings import settings
from ..themes.theme import Theme
from ..util.dataclasses import dataclass, field
from ..util.serialization import (
def standalone_docs_json(models: Sequence[Model | Document]) -> dict[ID, DocJson]:
    """

    """
    docs_json, _ = standalone_docs_json_and_render_items(models)
    return docs_json