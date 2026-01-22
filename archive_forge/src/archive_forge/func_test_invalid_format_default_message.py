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
def test_invalid_format_default_message(self):
    checker = FormatChecker(formats=())
    checker.checks('thing')(lambda value: False)
    schema = {'format': 'thing'}
    message = self.message_for(instance='bla', schema=schema, format_checker=checker)
    self.assertIn(repr('bla'), message)
    self.assertIn(repr('thing'), message)
    self.assertIn('is not a', message)