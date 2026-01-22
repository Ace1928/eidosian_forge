from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast
from fontTools.designspaceLib import (
def userRegionToDesignRegion(doc: DesignSpaceDocument, userRegion: Region) -> Region:
    designRegion = {}
    for name, value in userRegion.items():
        axis = doc.getAxis(name)
        if axis is None:
            raise DesignSpaceDocumentError(f"Cannot find axis named '{name}' for region.")
        if isinstance(value, (float, int)):
            designRegion[name] = axis.map_forward(value)
        else:
            designRegion[name] = Range(axis.map_forward(value.minimum), axis.map_forward(value.maximum), axis.map_forward(value.default))
    return designRegion