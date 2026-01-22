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
def buildDependencyCallList(self, idstr, newList, addMethod, removeMethod):
    oldList = self.getLastDependencyList(idstr)
    calls = []
    calls += [[addMethod, [wrapId(x)]] for x in newList if x not in oldList]
    calls += [[removeMethod, [wrapId(x)]] for x in oldList if x not in newList]
    self.setNewDependencyList(idstr, newList)
    return calls