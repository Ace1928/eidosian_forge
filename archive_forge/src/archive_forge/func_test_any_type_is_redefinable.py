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
def test_any_type_is_redefinable(self):
    """
        Sigh, because why not.
        """
    Crazy = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine('any', lambda checker, thing: isinstance(thing, int)))
    validator = Crazy({'type': 'any'})
    validator.validate(12)
    with self.assertRaises(exceptions.ValidationError):
        validator.validate('foo')