from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.utils import to_tuple
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.value.iterable import SequenceLiteralValue
from jedi.inference.helpers import is_string
def lambda_scoping_in_for_loop_sucks(lazy_value):
    return lambda: ValueSet(_resolve_forward_references(self._context_of_index, lazy_value.infer()))