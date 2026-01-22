from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast
from fontTools.designspaceLib import (
def getVFUserRegion(doc: DesignSpaceDocument, vf: VariableFontDescriptor) -> Region:
    vfUserRegion: Region = {}
    for axisSubset in vf.axisSubsets:
        axis = doc.getAxis(axisSubset.name)
        if axis is None:
            raise DesignSpaceDocumentError(f"Cannot find axis named '{axisSubset.name}' for variable font '{vf.name}'.")
        if hasattr(axisSubset, 'userMinimum'):
            axisSubset = cast(RangeAxisSubsetDescriptor, axisSubset)
            if not hasattr(axis, 'minimum'):
                raise DesignSpaceDocumentError(f"Cannot select a range over '{axis.name}' for variable font '{vf.name}' because it's a discrete axis, use only 'userValue' instead.")
            axis = cast(AxisDescriptor, axis)
            vfUserRegion[axis.name] = Range(max(axisSubset.userMinimum, axis.minimum), min(axisSubset.userMaximum, axis.maximum), axisSubset.userDefault or axis.default)
        else:
            axisSubset = cast(ValueAxisSubsetDescriptor, axisSubset)
            vfUserRegion[axis.name] = axisSubset.userValue
    for axis in doc.axes:
        if axis.name not in vfUserRegion:
            assert isinstance(axis.default, (int, float)), f"Axis '{axis.name}' has no valid default value."
            vfUserRegion[axis.name] = axis.default
    return vfUserRegion