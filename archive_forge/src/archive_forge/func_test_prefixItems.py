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
def test_prefixItems(self):
    schema = {'prefixItems': [{'type': 'string'}, {}, {}, {'maximum': 3}]}
    validator = validators.Draft202012Validator(schema)
    type_error, min_error = validator.iter_errors([1, 2, 'foo', 5])
    self.assertEqual((type_error.message, type_error.validator, type_error.validator_value, type_error.instance, type_error.absolute_path, type_error.schema, type_error.schema_path, type_error.json_path), ("1 is not of type 'string'", 'type', 'string', 1, deque([0]), {'type': 'string'}, deque(['prefixItems', 0, 'type']), '$[0]'))
    self.assertEqual((min_error.message, min_error.validator, min_error.validator_value, min_error.instance, min_error.absolute_path, min_error.schema, min_error.schema_path, min_error.json_path), ('5 is greater than the maximum of 3', 'maximum', 3, 5, deque([3]), {'maximum': 3}, deque(['prefixItems', 3, 'maximum']), '$[3]'))