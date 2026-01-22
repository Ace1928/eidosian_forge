import math
import secrets
import uuid as _uu
from typing import List
from typing import Optional
def uuid(self, name: Optional[str]=None, pad_length: Optional[int]=None) -> str:
    """
        Generate and return a UUID.

        If the name parameter is provided, set the namespace to the provided
        name and generate a UUID.
        """
    if pad_length is None:
        pad_length = self._length
    if name is None:
        u = _uu.uuid4()
    elif name.lower().startswith(('http://', 'https://')):
        u = _uu.uuid5(_uu.NAMESPACE_URL, name)
    else:
        u = _uu.uuid5(_uu.NAMESPACE_DNS, name)
    return self.encode(u, pad_length)