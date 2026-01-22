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
def test_omitted_arg(self):

    @njit(debug=True)
    def foo(missing=None):
        pass
    with override_config('DEBUGINFO_DEFAULT', 1):
        foo()
    metadata = self._get_metadata(foo, sig=(types.Omitted(None),))
    metadata_definition_map = self._get_metadata_map(metadata)
    tmp_disubr = []
    for md in metadata:
        if 'DISubroutineType' in md:
            tmp_disubr.append(md)
    self.assertEqual(len(tmp_disubr), 1)
    disubr = tmp_disubr.pop()
    disubr_matched = re.match('.*!DISubroutineType\\(types: ([!0-9]+)\\)$', disubr)
    self.assertIsNotNone(disubr_matched)
    disubr_groups = disubr_matched.groups()
    self.assertEqual(len(disubr_groups), 1)
    disubr_meta = disubr_groups[0]
    disubr_types = metadata_definition_map[disubr_meta]
    disubr_types_matched = re.match('!{(.*)}', disubr_types)
    self.assertIsNotNone(disubr_matched)
    disubr_types_groups = disubr_types_matched.groups()
    self.assertEqual(len(disubr_types_groups), 1)
    md_fn_arg = [x.strip() for x in disubr_types_groups[0].split(',')][-1]
    arg_ty = metadata_definition_map[md_fn_arg]
    expected_arg_ty = '^.*!DICompositeType\\(tag: DW_TAG_structure_type, name: "Anonymous struct \\({}\\)", elements: (![0-9]+), identifier: "{}"\\)'
    self.assertRegex(arg_ty, expected_arg_ty)
    md_base_ty = re.match(expected_arg_ty, arg_ty).groups()[0]
    base_ty = metadata_definition_map[md_base_ty]
    self.assertEqual(base_ty, '!{}')