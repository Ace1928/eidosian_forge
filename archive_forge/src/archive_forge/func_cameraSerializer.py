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
def cameraSerializer(parent, instance, objId, context, depth):
    return {'parent': context.getReferenceId(parent), 'id': objId, 'type': instance.GetClassName(), 'properties': {'focalPoint': instance.GetFocalPoint(), 'position': instance.GetPosition(), 'viewUp': instance.GetViewUp(), 'clippingRange': instance.GetClippingRange()}}