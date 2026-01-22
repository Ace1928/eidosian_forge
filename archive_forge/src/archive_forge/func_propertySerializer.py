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
def propertySerializer(parent, propObj, propObjId, context, depth):
    representation = propObj.GetRepresentation() if hasattr(propObj, 'GetRepresentation') else 2
    colorToUse = propObj.GetDiffuseColor() if hasattr(propObj, 'GetDiffuseColor') else [1, 1, 1]
    if representation == 1 and hasattr(propObj, 'GetColor'):
        colorToUse = propObj.GetColor()
    return {'parent': context.getReferenceId(parent), 'id': propObjId, 'type': propObj.GetClassName(), 'properties': {'representation': representation, 'diffuseColor': colorToUse, 'color': propObj.GetColor(), 'ambientColor': propObj.GetAmbientColor(), 'specularColor': propObj.GetSpecularColor(), 'edgeColor': propObj.GetEdgeColor(), 'ambient': propObj.GetAmbient(), 'diffuse': propObj.GetDiffuse(), 'specular': propObj.GetSpecular(), 'specularPower': propObj.GetSpecularPower(), 'opacity': propObj.GetOpacity(), 'interpolation': propObj.GetInterpolation(), 'edgeVisibility': 1 if propObj.GetEdgeVisibility() else 0, 'backfaceCulling': 1 if propObj.GetBackfaceCulling() else 0, 'frontfaceCulling': 1 if propObj.GetFrontfaceCulling() else 0, 'pointSize': propObj.GetPointSize(), 'lineWidth': propObj.GetLineWidth(), 'lighting': 1 if propObj.GetLighting() else 0}}