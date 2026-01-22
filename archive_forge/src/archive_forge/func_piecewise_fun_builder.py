import json
import re
import struct
import zipfile
import vtk
from .synchronizable_serializer import arrayTypesMapping
def piecewise_fun_builder(state, zf, register):
    instance = getattr(vtk, state['type'])()
    register.update({state['id']: instance})
    nodes = state['properties'].pop('nodes')
    set_properties(instance, state['properties'])
    for node in nodes:
        instance.AddPoint(*node)