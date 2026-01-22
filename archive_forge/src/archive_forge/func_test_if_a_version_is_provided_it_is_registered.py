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
def test_if_a_version_is_provided_it_is_registered(self):
    Validator = validators.create(meta_schema={'$id': 'something'}, version='my version')
    self.addCleanup(validators._META_SCHEMAS.pop, 'something')
    self.addCleanup(validators._VALIDATORS.pop, 'my version')
    self.assertEqual(Validator.__name__, 'MyVersionValidator')
    self.assertEqual(Validator.__qualname__, 'MyVersionValidator')