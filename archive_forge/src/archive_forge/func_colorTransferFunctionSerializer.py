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
def colorTransferFunctionSerializer(parent, instance, objId, context, depth):
    nodes = []
    for i in range(instance.GetSize()):
        node = [0, 0, 0, 0, 0, 0]
        instance.GetNodeValue(i, node)
        nodes.append(node)
    return {'parent': context.getReferenceId(parent), 'id': objId, 'type': instance.GetClassName(), 'properties': {'clamping': 1 if instance.GetClamping() else 0, 'colorSpace': instance.GetColorSpace(), 'hSVWrap': 1 if instance.GetHSVWrap() else 0, 'allowDuplicateScalars': 1 if instance.GetAllowDuplicateScalars() else 0, 'alpha': instance.GetAlpha(), 'vectorComponent': instance.GetVectorComponent(), 'vectorSize': instance.GetVectorSize(), 'vectorMode': instance.GetVectorMode(), 'indexedLookup': instance.GetIndexedLookup(), 'nodes': nodes}}