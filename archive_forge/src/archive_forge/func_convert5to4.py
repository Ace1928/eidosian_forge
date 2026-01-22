from __future__ import annotations
import itertools
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple, cast
from fontTools.designspaceLib import (
from fontTools.designspaceLib.statNames import StatNames, getStatNames
from fontTools.designspaceLib.types import (
def convert5to4(doc: DesignSpaceDocument) -> Dict[str, DesignSpaceDocument]:
    """Convert each variable font listed in this document into a standalone
    format 4 designspace. This can be used to compile all the variable fonts
    from a format 5 designspace using tools that only know about format 4.

    .. versionadded:: 5.0
    """
    vfs = {}
    for _location, subDoc in splitInterpolable(doc):
        for vfName, vfDoc in splitVariableFonts(subDoc):
            vfDoc.formatVersion = '4.1'
            vfs[vfName] = vfDoc
    return vfs