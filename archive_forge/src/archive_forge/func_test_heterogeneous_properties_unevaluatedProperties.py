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
def test_heterogeneous_properties_unevaluatedProperties(self):
    """
        Not valid deserialized JSON, but this should not blow up.
        """
    schema = {'properties': {'foo': {}}, 'unevaluatedProperties': False}
    message = self.message_for(instance={'foo': {}, 'a': 'baz', 37: 12}, schema=schema)
    self.assertEqual(message, "Unevaluated properties are not allowed (37, 'a' were unexpected)")