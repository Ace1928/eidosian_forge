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
def test_long_repr(self):
    Validator = validators.create(meta_schema={'$id': 'something'}, version='my version')
    self.addCleanup(validators._META_SCHEMAS.pop, 'something')
    self.addCleanup(validators._VALIDATORS.pop, 'my version')
    self.assertEqual(repr(Validator({'a': list(range(1000))})), "MyVersionValidator(schema={'a': [0, 1, 2, 3, 4, 5, ...]}, format_checker=None)")