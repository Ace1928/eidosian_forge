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
def lookupTableToColorTransferFunction(lookupTable):
    dataTable = lookupTable.GetTable()
    table = dataTableToList(dataTable)
    if table:
        ctf = vtkColorTransferFunction()
        tableRange = lookupTable.GetTableRange()
        points = linspace(*tableRange, num=len(table))
        for x, rgba in zip(points, table):
            ctf.AddRGBPoint(x, *[x / 255 for x in rgba[:3]])
        return ctf