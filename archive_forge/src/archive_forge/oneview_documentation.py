from __future__ import (absolute_import, division, print_function)
import abc
import collections
import json
import os  # noqa: F401, pylint: disable=unused-import
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._collections_compat import Mapping

        Recursively compares lists contents equivalence, ignoring types and element orders.
        Lists with same size are compared value by value after a sort,
        each element is converted to str before the comparison.
        :arg list first_resource: first list
        :arg list second_resource: second list
        :return: True when equal; False when different.
        