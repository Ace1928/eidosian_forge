from __future__ import absolute_import, print_function, division
import operator
from petl.compat import text_type
from petl.util.base import Table, asindices, Record
def normalize_constraints(constraints, flds):
    """
    This method renders local constraints such that return value is:
      * a list, not None
      * a list of dicts
      * a list of non-optional constraints or optional with defined field

    .. note:: We use a new variable 'local_constraints' because the constraints
              parameter may be a mutable collection, and we do not wish to
              cause side-effects by modifying it locally
    """
    local_constraints = constraints or []
    local_constraints = [dict(**c) for c in local_constraints]
    local_constraints = [c for c in local_constraints if c.get('field') in flds or not c.get('optional')]
    return local_constraints