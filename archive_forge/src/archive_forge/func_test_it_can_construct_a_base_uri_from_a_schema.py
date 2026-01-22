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
def test_it_can_construct_a_base_uri_from_a_schema(self):
    schema = {'id': 'foo'}
    resolver = validators._RefResolver.from_schema(schema, id_of=lambda schema: schema.get('id', ''))
    self.assertEqual(resolver.base_uri, 'foo')
    self.assertEqual(resolver.resolution_scope, 'foo')
    with resolver.resolving('') as resolved:
        self.assertEqual(resolved, schema)
    with resolver.resolving('#') as resolved:
        self.assertEqual(resolved, schema)
    with resolver.resolving('foo') as resolved:
        self.assertEqual(resolved, schema)
    with resolver.resolving('foo#') as resolved:
        self.assertEqual(resolved, schema)