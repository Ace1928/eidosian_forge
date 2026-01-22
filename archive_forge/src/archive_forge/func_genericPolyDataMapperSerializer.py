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
def genericPolyDataMapperSerializer(parent, mapper, mapperId, context, depth):
    instance = genericMapperSerializer(parent, mapper, mapperId, context, depth)
    if not instance:
        return
    instance['type'] = mapper.GetClassName()
    instance['properties'].update({'resolveCoincidentTopology': mapper.GetResolveCoincidentTopology(), 'renderTime': mapper.GetRenderTime(), 'arrayAccessMode': 1, 'scalarRange': mapper.GetScalarRange(), 'useLookupTableScalarRange': 1 if mapper.GetUseLookupTableScalarRange() else 0, 'scalarVisibility': mapper.GetScalarVisibility(), 'colorByArrayName': retrieveArrayName(instance, mapper.GetScalarMode()), 'colorMode': mapper.GetColorMode(), 'scalarMode': mapper.GetScalarMode(), 'interpolateScalarsBeforeMapping': 1 if mapper.GetInterpolateScalarsBeforeMapping() else 0})
    return instance