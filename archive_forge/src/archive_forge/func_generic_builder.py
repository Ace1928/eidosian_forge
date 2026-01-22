import json
import re
import struct
import zipfile
import vtk
from .synchronizable_serializer import arrayTypesMapping
def generic_builder(state, zf, register=None):
    if register is None:
        register = {}
    instance = getattr(vtk, state['type'])()
    register.update({state['id']: instance})
    set_properties(instance, state['properties'])
    dependencies = state.get('dependencies', None)
    if dependencies:
        for dep in dependencies:
            builder = TYPE_HANDLERS[dep['type']]
            if builder:
                builder(dep, zf, register)
            else:
                print(f'No builder for {dep['type']}')
    calls = state.get('calls', None)
    if calls:
        for call in calls:
            args = []
            skip = False
            for arg in call[1]:
                try:
                    extract_instance = WRAP_ID_RE.findall(arg)[0]
                    args.append(register[extract_instance])
                except (IndexError, TypeError):
                    args.append(arg)
                except KeyError:
                    skip = True
            if skip:
                continue
            if capitalize(call[0]) not in METHODS_RENAME:
                method = capitalize(call[0])
            else:
                method = METHODS_RENAME[capitalize(call[0])]
            if method is None:
                continue
            if method == 'SetInputData' and len(args) == 2:
                getattr(instance, method + 'Object')(*args[::-1])
            else:
                getattr(instance, method)(*args)
    arrays = state.get('arrays', None)
    if arrays:
        for array_meta in arrays:
            vtk_array = ARRAY_TYPES[array_meta['dataType']]()
            fill_array(vtk_array, array_meta, zf)
            location = instance if 'location' not in array_meta else getattr(instance, 'Get' + capitalize(array_meta['location']))()
            getattr(location, capitalize(array_meta['registration']))(vtk_array)
    return instance