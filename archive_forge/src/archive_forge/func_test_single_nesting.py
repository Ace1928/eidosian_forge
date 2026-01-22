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
def test_single_nesting(self):
    instance = {'foo': 2, 'bar': [1], 'baz': 15, 'quux': 'spam'}
    schema = {'properties': {'foo': {'type': 'string'}, 'bar': {'minItems': 2}, 'baz': {'maximum': 10, 'enum': [2, 4, 6, 8]}}}
    validator = validators.Draft3Validator(schema)
    errors = validator.iter_errors(instance)
    e1, e2, e3, e4 = sorted_errors(errors)
    self.assertEqual(e1.path, deque(['bar']))
    self.assertEqual(e2.path, deque(['baz']))
    self.assertEqual(e3.path, deque(['baz']))
    self.assertEqual(e4.path, deque(['foo']))
    self.assertEqual(e1.relative_path, deque(['bar']))
    self.assertEqual(e2.relative_path, deque(['baz']))
    self.assertEqual(e3.relative_path, deque(['baz']))
    self.assertEqual(e4.relative_path, deque(['foo']))
    self.assertEqual(e1.absolute_path, deque(['bar']))
    self.assertEqual(e2.absolute_path, deque(['baz']))
    self.assertEqual(e3.absolute_path, deque(['baz']))
    self.assertEqual(e4.absolute_path, deque(['foo']))
    self.assertEqual(e1.json_path, '$.bar')
    self.assertEqual(e2.json_path, '$.baz')
    self.assertEqual(e3.json_path, '$.baz')
    self.assertEqual(e4.json_path, '$.foo')
    self.assertEqual(e1.validator, 'minItems')
    self.assertEqual(e2.validator, 'enum')
    self.assertEqual(e3.validator, 'maximum')
    self.assertEqual(e4.validator, 'type')