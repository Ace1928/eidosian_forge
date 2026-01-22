from __future__ import annotations
from typing import Dict, List, Union
import fontTools.otlLib.builder
from fontTools.designspaceLib import (
from fontTools.designspaceLib.types import Region, getVFUserRegion, locationInRegion
from fontTools.ttLib import TTFont
def getStatLocations(doc: DesignSpaceDocument, userRegion: Region) -> List[Dict]:
    """Return a list of location dicts suitable for use as the ``locations``
    argument to :func:`fontTools.otlLib.builder.buildStatTable()`.

    .. versionadded:: 5.0
    """
    axesByName = {axis.name: axis for axis in doc.axes}
    return [dict(name={'en': label.name, **label.labelNames}, location={axesByName[name].tag: value for name, value in label.getFullUserLocation(doc).items()}, flags=_labelToFlags(label)) for label in doc.locationLabels if locationInRegion(label.getFullUserLocation(doc), userRegion)]