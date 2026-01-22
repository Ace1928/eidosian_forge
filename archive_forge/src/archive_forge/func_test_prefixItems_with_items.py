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
def test_prefixItems_with_items(self):
    schema = {'items': {'type': 'string'}, 'prefixItems': [{}]}
    validator = validators.Draft202012Validator(schema)
    e1, e2 = validator.iter_errors(['foo', 2, 'bar', 4, 'baz'])
    self.assertEqual((e1.message, e1.validator, e1.validator_value, e1.instance, e1.absolute_path, e1.schema, e1.schema_path, e1.json_path), ("2 is not of type 'string'", 'type', 'string', 2, deque([1]), {'type': 'string'}, deque(['items', 'type']), '$[1]'))
    self.assertEqual((e2.message, e2.validator, e2.validator_value, e2.instance, e2.absolute_path, e2.schema, e2.schema_path, e2.json_path), ("4 is not of type 'string'", 'type', 'string', 4, deque([3]), {'type': 'string'}, deque(['items', 'type']), '$[3]'))