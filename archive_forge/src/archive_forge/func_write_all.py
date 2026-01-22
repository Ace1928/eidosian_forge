import builtins
import codecs
import enum
import io
import json
import os
import types
import typing
from typing import (
import attr
def write_all(self, iterable: Iterable[Any]) -> int:
    """
        Encode and write multiple objects.

        :param iterable: an iterable of objects
        :return: number of characters or bytes written
        """
    return sum((self.write(obj) for obj in iterable))