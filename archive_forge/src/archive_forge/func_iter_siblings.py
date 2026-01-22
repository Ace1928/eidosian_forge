from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
def iter_siblings(self):
    """ Yield all siblings of this DIE
        """
    parent = self.get_parent()
    if parent:
        for sibling in parent.iter_children():
            if sibling is not self:
                yield sibling
    else:
        raise StopIteration()