from pyrsistent._checked_types import (
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
def check_global_invariants(subject, invariants):
    error_codes = tuple((error_code for is_ok, error_code in (invariant(subject) for invariant in invariants) if not is_ok))
    if error_codes:
        raise InvariantException(error_codes, (), 'Global invariant failed')