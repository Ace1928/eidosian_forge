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
def test_it_validates_formats_if_a_checker_is_provided(self):
    checker = FormatChecker()
    bad = ValueError('Bad!')

    @checker.checks('foo', raises=ValueError)
    def check(value):
        if value == 'good':
            return True
        elif value == 'bad':
            raise bad
        else:
            self.fail(f"What is {value}? [Baby Don't Hurt Me]")
    validator = self.Validator({'format': 'foo'}, format_checker=checker)
    validator.validate('good')
    with self.assertRaises(exceptions.ValidationError) as cm:
        validator.validate('bad')
    self.assertIs(cm.exception.cause, bad)