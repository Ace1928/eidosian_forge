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
def imagePropertySerializer(parent, propObj, propObjId, context, depth):
    calls = []
    dependencies = []
    lookupTable = propObj.GetLookupTable()
    if lookupTable:
        ctfun = lookupTableToColorTransferFunction(lookupTable)
        ctfunId = context.getReferenceId(ctfun)
        ctfunInstance = serializeInstance(propObj, ctfun, ctfunId, context, depth + 1)
        if ctfunInstance:
            dependencies.append(ctfunInstance)
            calls.append(['setRGBTransferFunction', [wrapId(ctfunId)]])
    return {'parent': context.getReferenceId(parent), 'id': propObjId, 'type': propObj.GetClassName(), 'properties': {'interpolationType': propObj.GetInterpolationType(), 'colorWindow': propObj.GetColorWindow(), 'colorLevel': propObj.GetColorLevel(), 'ambient': propObj.GetAmbient(), 'diffuse': propObj.GetDiffuse(), 'opacity': propObj.GetOpacity()}, 'dependencies': dependencies, 'calls': calls}