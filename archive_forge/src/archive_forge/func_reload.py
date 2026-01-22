from ... import errors, tests, transport
from .. import index as _mod_index
def reload():
    reload_counter[0] += 1
    new_indices = [idx3]
    if idx._indices == new_indices:
        reload_counter[2] += 1
        return False
    reload_counter[1] += 1
    idx._indices[:] = new_indices
    return True