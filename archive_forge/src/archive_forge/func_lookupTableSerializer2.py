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
def lookupTableSerializer2(parent, lookupTable, lookupTableId, context, depth):
    ctf = lookupTableToColorTransferFunction(lookupTable)
    if ctf:
        return colorTransferFunctionSerializer(parent, ctf, lookupTableId, context, depth)