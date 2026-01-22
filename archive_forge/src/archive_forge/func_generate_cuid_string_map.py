import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
@staticmethod
def generate_cuid_string_map(block, ctype=None, descend_into=True, repr_version=2):

    def _record_indexed_object_cuid_strings_v1(obj, cuid_str):
        _unknown = lambda x: '?' + str(x)
        for idx, data in obj.items():
            if idx.__class__ is tuple and len(idx) > 1:
                cuid_strings[data] = cuid_str + ':' + ','.join((ComponentUID._repr_v1_map.get(x.__class__, _unknown)(x) for x in idx))
            else:
                cuid_strings[data] = cuid_str + ':' + ComponentUID._repr_v1_map.get(idx.__class__, _unknown)(idx)

    def _record_indexed_object_cuid_strings_v2(obj, cuid_str):
        for idx, data in obj.items():
            cuid_strings[data] = cuid_str + _index_repr(idx)
    _record_indexed_object_cuid_strings = {1: _record_indexed_object_cuid_strings_v1, 2: _record_indexed_object_cuid_strings_v2}[repr_version]
    _record_name = {1: str, 2: _name_repr}[repr_version]
    model = block.model()
    cuid_strings = ComponentMap()
    cuid_strings[block] = ComponentUID(block).get_repr(repr_version)
    for blk in block.block_data_objects(descend_into=descend_into):
        if blk not in cuid_strings:
            blk_comp = blk.parent_component()
            cuid_str = _record_name(blk_comp.local_name)
            blk_pblk = blk_comp.parent_block()
            if blk_pblk is not model:
                cuid_str = cuid_strings[blk_pblk] + '.' + cuid_str
            cuid_strings[blk_comp] = cuid_str
            if blk_comp.is_indexed():
                _record_indexed_object_cuid_strings(blk_comp, cuid_str)
        for obj in blk.component_objects(ctype=ctype, descend_into=False):
            cuid_str = _record_name(obj.local_name)
            if blk is not model:
                cuid_str = cuid_strings[blk] + '.' + cuid_str
            cuid_strings[obj] = cuid_str
            if obj.is_indexed():
                _record_indexed_object_cuid_strings(obj, cuid_str)
    return cuid_strings