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
def test_it_does_not_retrieve_schema_urls_from_the_network(self):
    ref = validators.Draft3Validator.META_SCHEMA['id']
    with mock.patch.object(self.resolver, 'resolve_remote') as patched:
        with self.resolver.resolving(ref) as resolved:
            pass
    self.assertEqual(resolved, validators.Draft3Validator.META_SCHEMA)
    self.assertFalse(patched.called)