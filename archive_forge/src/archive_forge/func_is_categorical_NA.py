import numpy as np
from patsy import PatsyError
from patsy.util import (safe_isnan, safe_scalar_isnan,
def is_categorical_NA(self, obj):
    """Return True if `obj` is a categorical NA value.

        Note that here `obj` is a single scalar value."""
    if 'NaN' in self.NA_types and safe_scalar_isnan(obj):
        return True
    if 'None' in self.NA_types and obj is None:
        return True
    return False