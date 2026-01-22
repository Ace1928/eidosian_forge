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
def test_it_creates_a_ref_resolver_if_not_provided(self):
    with self.assertWarns(DeprecationWarning):
        resolver = self.Validator({}).resolver
    self.assertIsInstance(resolver, validators._RefResolver)