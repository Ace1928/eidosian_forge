from __future__ import annotations
from typing import Any
from unittest import TestCase
from traitlets import TraitError
def test_good_values(self) -> None:
    if hasattr(self, '_good_values'):
        for value in self._good_values:
            self.assign(value)
            self.assertEqual(self.obj.value, self.coerce(value))