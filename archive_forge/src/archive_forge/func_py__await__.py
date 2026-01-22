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
def py__await__(self):
    await_value_set = self.py__getattribute__('__await__')
    if not await_value_set:
        debug.warning('Tried to run __await__ on value %s', self)
    return await_value_set.execute_with_values()