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
def merge_types_of_iterate(self, contextualized_node=None, is_async=False):
    return ValueSet.from_sets((lazy_value.infer() for lazy_value in self.iterate(contextualized_node, is_async)))