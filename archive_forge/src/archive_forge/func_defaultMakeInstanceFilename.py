from __future__ import annotations
import itertools
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple, cast
from fontTools.designspaceLib import (
from fontTools.designspaceLib.statNames import StatNames, getStatNames
from fontTools.designspaceLib.types import (
def defaultMakeInstanceFilename(doc: DesignSpaceDocument, instance: InstanceDescriptor, statNames: StatNames) -> str:
    """Default callable to synthesize an instance filename
    when makeNames=True, for instances that don't specify an instance name
    in the designspace. This part of the name generation can be overriden
    because it's not specified by the STAT table.
    """
    familyName = instance.familyName or statNames.familyNames.get('en')
    styleName = instance.styleName or statNames.styleNames.get('en')
    return f'{familyName}-{styleName}.ttf'