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
def test_ref_sibling(self):
    schema = {'$defs': {'foo': {'required': ['bar']}}, 'properties': {'aprop': {'$ref': '#/$defs/foo', 'required': ['baz']}}}
    validator = validators.Draft202012Validator(schema)
    e1, e2 = validator.iter_errors({'aprop': {}})
    self.assertEqual((e1.message, e1.validator, e1.validator_value, e1.instance, e1.absolute_path, e1.schema, e1.schema_path, e1.relative_schema_path, e1.json_path), ("'bar' is a required property", 'required', ['bar'], {}, deque(['aprop']), {'required': ['bar']}, deque(['properties', 'aprop', 'required']), deque(['properties', 'aprop', 'required']), '$.aprop'))
    self.assertEqual((e2.message, e2.validator, e2.validator_value, e2.instance, e2.absolute_path, e2.schema, e2.schema_path, e2.relative_schema_path, e2.json_path), ("'baz' is a required property", 'required', ['baz'], {}, deque(['aprop']), {'$ref': '#/$defs/foo', 'required': ['baz']}, deque(['properties', 'aprop', 'required']), deque(['properties', 'aprop', 'required']), '$.aprop'))