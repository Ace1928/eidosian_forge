from __future__ import annotations
from collections import deque, namedtuple
from contextlib import contextmanager
from decimal import Decimal
from io import BytesIO
from typing import Any
from unittest import TestCase, mock
from urllib.request import pathname2url
import json
import os
import sys
import tempfile
import warnings
from attrs import define, field
from referencing.jsonschema import DRAFT202012
import referencing.exceptions
from jsonschema import (
def test_evolve_with_subclass(self):
    """
        Subclassing validators isn't supported public API, but some users have
        done it, because we don't actually error entirely when it's done :/

        We need to deprecate doing so first to help as many of these users
        ensure they can move to supported APIs, but this test ensures that in
        the interim, we haven't broken those users.
        """
    with self.assertWarns(DeprecationWarning):

        @define
        class OhNo(self.Validator):
            foo = field(factory=lambda: [1, 2, 3])
            _bar = field(default=37)
    validator = OhNo({}, bar=12)
    self.assertEqual(validator.foo, [1, 2, 3])
    new = validator.evolve(schema={'type': 'integer'})
    self.assertEqual(new.foo, [1, 2, 3])
    self.assertEqual(new._bar, 12)