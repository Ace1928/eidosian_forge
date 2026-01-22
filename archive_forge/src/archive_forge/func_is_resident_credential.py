import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
@property
def is_resident_credential(self) -> bool:
    return self._is_resident_credential