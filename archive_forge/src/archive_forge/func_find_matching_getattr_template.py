from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def find_matching_getattr_template(self, typ, attr):
    templates = list(self._get_attribute_templates(typ))
    from numba.core.target_extension import get_local_target
    target_hw = get_local_target(self)
    order = order_by_target_specificity(target_hw, templates, fnkey=attr)
    for template in order:
        return_type = template.resolve(typ, attr)
        if return_type is not None:
            return {'template': template, 'return_type': return_type}