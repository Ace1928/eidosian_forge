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
def test_schema_error_message(self):
    with self.assertRaises(exceptions.SchemaError) as e:
        validators.validate(12, {'type': 12})
    self.assertRegex(str(e.exception), "(?s)Failed validating '.*' in metaschema.*On schema")