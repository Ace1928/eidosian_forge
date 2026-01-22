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
def getLastDependencyList(self, idstr):
    lastDeps = []
    if idstr in self.lastDependenciesMapping and (not self.ingoreLastDependencies):
        lastDeps = self.lastDependenciesMapping[idstr]
    return lastDeps