import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
@property
def attributes_schema(self):
    """A set of the valid top-level attribute names.

        This is provided for backwards-compatibility for functions that require
        a container with all of the valid attribute names in order to validate
        the template. Other operations on it are invalid because we don't
        actually have access to the attributes schema here; hence we return a
        set instead of a dict.
        """
    return set(self._res_data().attribute_names())