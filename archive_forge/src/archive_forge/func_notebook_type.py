from __future__ import annotations
import logging # isort:skip
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from ..core.types import PathLike
from ..document import Document
from ..resources import Resources, ResourcesMode
@notebook_type.setter
def notebook_type(self, notebook_type: NotebookType) -> None:
    """ Notebook type, acceptable values are 'jupyter' as well as any names
        defined by external notebook hooks that have been installed.

        """
    if notebook_type is None or not isinstance(notebook_type, str):
        raise ValueError('Notebook type must be a string')
    self._notebook_type = cast('NotebookType', notebook_type.lower())