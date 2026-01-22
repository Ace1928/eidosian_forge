import copy
import re
from aiokeydb.v1.commands.helpers import nativestr
def bulk_of_jsons(d):
    """Replace serialized JSON values with objects in a
    bulk array response (list).
    """

    def _f(b):
        for index, item in enumerate(b):
            if item is not None:
                b[index] = d(item)
        return b
    return _f