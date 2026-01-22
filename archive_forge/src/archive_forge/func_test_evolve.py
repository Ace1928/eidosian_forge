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
def test_evolve(self):
    schema, format_checker = ({'type': 'integer'}, FormatChecker())
    original = self.Validator(schema, format_checker=format_checker)
    new = original.evolve(schema={'type': 'string'}, format_checker=self.Validator.FORMAT_CHECKER)
    expected = self.Validator({'type': 'string'}, format_checker=self.Validator.FORMAT_CHECKER, _resolver=new._resolver)
    self.assertEqual(new, expected)
    self.assertNotEqual(new, original)