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
def test_pointer_within_schema_with_different_id(self):
    """
        See #1085.
        """
    schema = validators.Draft7Validator.META_SCHEMA
    one = validators._RefResolver('', schema)
    validator = validators.Draft7Validator(schema, resolver=one)
    self.assertFalse(validator.is_valid({'maxLength': 'foo'}))
    another = {'allOf': [{'$ref': validators.Draft7Validator.META_SCHEMA['$id']}]}
    two = validators._RefResolver('', another)
    validator = validators.Draft7Validator(another, resolver=two)
    self.assertFalse(validator.is_valid({'maxLength': 'foo'}))