from functools import reduce
from operator import add
from itertools import zip_longest
from parso.python.tree import Name
from jedi import debug
from jedi.parser_utils import clean_scope_docstring
from jedi.inference.helpers import SimpleGetItemNotFound
from jedi.inference.utils import safe_property
from jedi.inference.cache import inference_state_as_method_param_cache
from jedi.cache import memoize_method
def try_merge(self, function_name):
    value_set = self.__class__([])
    for c in self._set:
        try:
            method = getattr(c, function_name)
        except AttributeError:
            pass
        else:
            value_set |= method()
    return value_set