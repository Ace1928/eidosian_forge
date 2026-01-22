import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
def parse_from_attribute(self, attr, dwarf_version, die=None):
    """ Parses a DIE attribute and returns either a LocationExpr or
            a list.
        """
    if self.attribute_has_location(attr, dwarf_version):
        if self._attribute_has_loc_expr(attr, dwarf_version):
            return LocationExpr(attr.value)
        elif self._attribute_has_loc_list(attr, dwarf_version):
            return self.location_lists.get_location_list_at_offset(attr.value, die)
    else:
        raise ValueError('Attribute does not have location information')