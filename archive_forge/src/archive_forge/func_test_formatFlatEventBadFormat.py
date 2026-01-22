import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_formatFlatEventBadFormat(self) -> None:
    """
        If the format string is invalid, an error is produced.
        """
    event1 = dict(log_format='strrepr: {string!X}', string='hello')
    flattenEvent(event1)
    event2 = json.loads(json.dumps(event1))
    self.assertTrue(formatEvent(event2).startswith('Unable to format event'))