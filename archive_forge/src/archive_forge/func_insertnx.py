from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function
def insertnx(self, key, items, capacity=None, nocreate=None):
    """
        Add multiple `items` to a Cuckoo Filter `key` only if they do not exist yet,
        allowing the filter to be created with a custom `capacity` if it does not yet exist.
        `items` must be provided as a list.
        For more information see `CF.INSERTNX <https://redis.io/commands/cf.insertnx>`_.
        """
    params = [key]
    self.append_capacity(params, capacity)
    self.append_no_create(params, nocreate)
    self.append_items(params, items)
    return self.execute_command(CF_INSERTNX, *params)