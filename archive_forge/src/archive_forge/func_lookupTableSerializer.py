import base64
import hashlib
import io
import struct
import time
import zipfile
from vtk.vtkCommonCore import vtkTypeInt32Array, vtkTypeUInt32Array
from vtk.vtkCommonDataModel import vtkDataObject
from vtk.vtkFiltersGeometry import (
from vtk.vtkRenderingCore import vtkColorTransferFunction
from .enums import TextPosition
def lookupTableSerializer(parent, lookupTable, lookupTableId, context, depth):
    arrays = []
    lookupTableRange = lookupTable.GetRange()
    lookupTableHueRange = [0.5, 0]
    if hasattr(lookupTable, 'GetHueRange'):
        try:
            lookupTable.GetHueRange(lookupTableHueRange)
        except Exception:
            pass
    lutSatRange = lookupTable.GetSaturationRange()
    if lookupTable.GetTable():
        arrayMeta = getArrayDescription(lookupTable.GetTable(), context)
        if arrayMeta:
            arrayMeta['registration'] = 'setTable'
            arrays.append(arrayMeta)
    return {'parent': context.getReferenceId(parent), 'id': lookupTableId, 'type': lookupTable.GetClassName(), 'properties': {'numberOfColors': lookupTable.GetNumberOfColors(), 'valueRange': lookupTableRange, 'range': lookupTableRange, 'hueRange': lookupTableHueRange, 'saturationRange': lutSatRange, 'nanColor': lookupTable.GetNanColor(), 'belowRangeColor': lookupTable.GetBelowRangeColor(), 'aboveRangeColor': lookupTable.GetAboveRangeColor(), 'useAboveRangeColor': True if lookupTable.GetUseAboveRangeColor() else False, 'useBelowRangeColor': True if lookupTable.GetUseBelowRangeColor() else False, 'alpha': lookupTable.GetAlpha(), 'vectorSize': lookupTable.GetVectorSize(), 'vectorComponent': lookupTable.GetVectorComponent(), 'vectorMode': lookupTable.GetVectorMode(), 'indexedLookup': lookupTable.GetIndexedLookup()}, 'arrays': arrays}