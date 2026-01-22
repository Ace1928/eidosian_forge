from collections import namedtuple
import inspect
import re
import numpy as np
import math
from textwrap import dedent
import unittest
import warnings
from numba.tests.support import (TestCase, override_config,
from numba import jit, njit
from numba.core import types
from numba.core.datamodel import default_manager
from numba.core.errors import NumbaDebugInfoWarning
import llvmlite.binding as llvm
def test_numeric_scalars(self):
    """ Tests that dwarf info is correctly emitted for numeric scalars."""
    DI = namedtuple('DI', 'name bits encoding')
    type_infos = {np.float32: DI('float32', 32, 'DW_ATE_float'), np.float64: DI('float64', 64, 'DW_ATE_float'), np.int8: DI('int8', 8, 'DW_ATE_signed'), np.int16: DI('int16', 16, 'DW_ATE_signed'), np.int32: DI('int32', 32, 'DW_ATE_signed'), np.int64: DI('int64', 64, 'DW_ATE_signed'), np.uint8: DI('uint8', 8, 'DW_ATE_unsigned'), np.uint16: DI('uint16', 16, 'DW_ATE_unsigned'), np.uint32: DI('uint32', 32, 'DW_ATE_unsigned'), np.uint64: DI('uint64', 64, 'DW_ATE_unsigned'), np.complex64: DI('complex64', 64, 'DW_TAG_structure_type'), np.complex128: DI('complex128', 128, 'DW_TAG_structure_type')}
    for ty, dwarf_info in type_infos.items():

        @njit(debug=True)
        def foo():
            a = ty(10)
            return a
        metadata = self._get_metadata(foo, sig=())
        metadata_definition_map = self._get_metadata_map(metadata)
        for k, v in metadata_definition_map.items():
            if 'DILocalVariable(name: "a"' in v:
                lvar = metadata_definition_map[k]
                break
        else:
            assert 0, "missing DILocalVariable 'a'"
        type_marker = re.match('.*type: (![0-9]+).*', lvar).groups()[0]
        type_decl = metadata_definition_map[type_marker]
        if 'DW_ATE' in dwarf_info.encoding:
            expected = f'!DIBasicType(name: "{dwarf_info.name}", size: {dwarf_info.bits}, encoding: {dwarf_info.encoding})'
            self.assertEqual(type_decl, expected)
        else:
            raw_flt = 'float' if dwarf_info.bits == 64 else 'double'
            expected = f'distinct !DICompositeType(tag: {dwarf_info.encoding}, name: "{dwarf_info.name} ({{{raw_flt}, {raw_flt}}})", size: {dwarf_info.bits}'
            self.assertIn(expected, type_decl)