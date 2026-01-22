import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_keyFlattening(self) -> None:
    """
        Test that L{KeyFlattener.flatKey} returns the expected keys for format
        fields.
        """

    def keyFromFormat(format: str) -> str:
        for literalText, fieldName, formatSpec, conversion in aFormatter.parse(format):
            assert fieldName is not None
            return KeyFlattener().flatKey(fieldName, formatSpec, conversion)
        assert False, 'Unable to derive key from format: {format}'
    try:
        self.assertEqual(keyFromFormat('{}'), '!:')
    except ValueError:
        raise
    self.assertEqual(keyFromFormat('{foo}'), 'foo!:')
    self.assertEqual(keyFromFormat('{foo!s}'), 'foo!s:')
    self.assertEqual(keyFromFormat('{foo!r}'), 'foo!r:')
    self.assertEqual(keyFromFormat('{foo:%s}'), 'foo!:%s')
    self.assertEqual(keyFromFormat('{foo:!}'), 'foo!:!')
    self.assertEqual(keyFromFormat('{foo::}'), 'foo!::')
    self.assertEqual(keyFromFormat('{foo!s:%s}'), 'foo!s:%s')
    self.assertEqual(keyFromFormat('{foo!s:!}'), 'foo!s:!')
    self.assertEqual(keyFromFormat('{foo!s::}'), 'foo!s::')
    sameFlattener = KeyFlattener()
    (literalText, fieldName, formatSpec, conversion), = aFormatter.parse('{x}')
    assert fieldName is not None
    self.assertEqual(sameFlattener.flatKey(fieldName, formatSpec, conversion), 'x!:')
    self.assertEqual(sameFlattener.flatKey(fieldName, formatSpec, conversion), 'x!:/2')