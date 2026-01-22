import collections
from llvmlite.ir import context, values, types, _utils
def get_unique_name(self, name=''):
    """
        Get a unique global name with the following *name* hint.
        """
    return self.scope.deduplicate(name)