from .array import ARRAY
from .array import array as _pg_array
from .operators import ASTEXT
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import DELETE_PATH
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from .operators import JSONPATH_ASTEXT
from .operators import PATH_EXISTS
from .operators import PATH_MATCH
from ... import types as sqltypes
from ...sql import cast
def contained_by(self, other):
    """Boolean expression.  Test if keys are a proper subset of the
            keys of the argument jsonb expression.
            """
    return self.operate(CONTAINED_BY, other, result_type=sqltypes.Boolean)