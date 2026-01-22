import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def test_formatFlatEventWithMutatedFields(self) -> None:
    """
        L{formatEvent} will prefer the stored C{str()} or C{repr()} value for
        an object, in case the other version.
        """

    class Unpersistable:
        """
            Unpersitable object.
            """
        destructed = False

        def selfDestruct(self) -> None:
            """
                Self destruct.
                """
            self.destructed = True

        def __repr__(self) -> str:
            if self.destructed:
                return 'post-serialization garbage'
            else:
                return 'un-persistable'
    up = Unpersistable()
    event1 = dict(log_format='unpersistable: {unpersistable}', unpersistable=up)
    flattenEvent(event1)
    up.selfDestruct()
    self.assertEqual(formatEvent(event1), 'unpersistable: un-persistable')