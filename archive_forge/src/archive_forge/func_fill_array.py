import json
import re
import struct
import zipfile
import vtk
from .synchronizable_serializer import arrayTypesMapping
def fill_array(vtk_arr, state, zf):
    vtk_arr.SetNumberOfComponents(state['numberOfComponents'])
    vtk_arr.SetNumberOfTuples(state['size'] // state['numberOfComponents'])
    data = zf.read('data/%s' % state['hash'])
    dataType = arrayTypesMapping[vtk_arr.GetDataType()]
    elementSize = struct.calcsize(dataType)
    if vtk_arr.GetDataType() == 12:
        import numpy as np
        data = np.frombuffer(data, dtype=np.uint32).astype(np.uint64).tobytes()
        elementSize = 8
    vtk_arr.SetVoidArray(data, len(data) // elementSize, 1)
    vtk_arr._reference = data