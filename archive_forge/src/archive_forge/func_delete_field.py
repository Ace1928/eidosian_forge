from __future__ import unicode_literals
from collections import deque
import copy
import functools
import re
def delete_field(request, field):
    """Delete the value of a field from a given dictionary.

    Args:
        request (dict | Message): A dictionary object or a Message.
        field (str): The key to the request in dot notation.
    """
    parts = deque(field.split('.'))
    while len(parts) > 1:
        part = parts.popleft()
        if not isinstance(request, dict):
            if hasattr(request, part):
                request = getattr(request, part, None)
            else:
                return
        else:
            request = request.get(part)
    part = parts.popleft()
    if not isinstance(request, dict):
        if hasattr(request, part):
            request.ClearField(part)
        else:
            return
    else:
        request.pop(part, None)