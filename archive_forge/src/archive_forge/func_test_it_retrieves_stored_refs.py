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
def test_it_retrieves_stored_refs(self):
    with self.resolver.resolving(self.stored_uri) as resolved:
        self.assertIs(resolved, self.stored_schema)
    self.resolver.store['cached_ref'] = {'foo': 12}
    with self.resolver.resolving('cached_ref#/foo') as resolved:
        self.assertEqual(resolved, 12)