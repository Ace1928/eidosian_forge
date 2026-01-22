import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def keyFromFormat(format: str) -> str:
    for literalText, fieldName, formatSpec, conversion in aFormatter.parse(format):
        assert fieldName is not None
        return KeyFlattener().flatKey(fieldName, formatSpec, conversion)
    assert False, 'Unable to derive key from format: {format}'