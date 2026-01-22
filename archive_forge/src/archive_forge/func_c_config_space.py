import ctypes
from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call
def c_config_space(x):
    """constructor for ConfigSpace"""
    ret = CConfigSpace()
    ret.entity_map_key = c_str_array(x._entity_map.keys())
    ret.entity_map_val = c_array(COtherOptionEntity, [c_other_option_entity(e) for e in x._entity_map.values()])
    ret.entity_map_size = len(x._entity_map)
    ret.space_map_key = c_str_array(x.space_map.keys())
    ret.space_map_val = c_array(COtherOptionSpace, [c_other_option_space(v) for v in x.space_map.values()])
    ret.space_map_size = len(x.space_map)
    return ret