import datetime
import difflib
import functools
import inspect
import json
import os
import re
import tempfile
import threading
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch._dynamo
import torch.utils._pytree as pytree
from torch._dynamo.utils import clone_input
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch._utils_internal import get_file_path_2
from torch.overrides import TorchFunctionMode
from torch.testing._internal.optests import (
def generate_tag_tests(testcase, failures_dict, additional_decorators):

    def generate_test(qualname, definitely_not_pt2_compliant, xfailed_tests):

        def inner(self):
            try:
                op = torch._library.utils.lookup_op(qualname)
            except AttributeError as e:
                raise unittest.SkipTest(f"Can't import operator {qualname}") from e
            op_marked_as_compliant = torch.Tag.pt2_compliant_tag in op.tags
            if not op_marked_as_compliant:
                return
            if not definitely_not_pt2_compliant:
                return
            raise AssertionError(f"op '{qualname}' was tagged with torch.Tag.pt2_compliant_tag but it failed some of the generated opcheck tests ({xfailed_tests}). This may lead to silent correctness issues, please fix this.")
        return inner
    for qualname, test_dict in failures_dict.data.items():
        xfailed_tests = [test for test, status_dict in test_dict.items() if 'test_aot_dispatch_static' not in test and status_dict['status'] == 'xfail']
        definitely_not_pt2_compliant = len(xfailed_tests) > 0
        generated = generate_test(qualname, definitely_not_pt2_compliant, xfailed_tests)
        mangled_qualname = qualname.replace('::', '_').replace('.', '_')
        test_name = 'test_pt2_compliant_tag_' + mangled_qualname
        if test_name in additional_decorators:
            for decorator in additional_decorators[test_name]:
                generated = decorator(generated)
        if hasattr(testcase, test_name):
            raise RuntimeError(f'Tried to generate a test named {test_name}, but it exists already. This could be because of a name collision (where we generated two tests with the same name), or where we generated a test with the same name as an existing test.')
        setattr(testcase, test_name, generated)