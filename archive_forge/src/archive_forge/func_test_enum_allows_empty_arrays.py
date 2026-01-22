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
def test_enum_allows_empty_arrays(self):
    """
        Technically, all the spec says is they SHOULD have elements, not MUST.

        (As of Draft 6. Previous drafts do say MUST).

        See #529.
        """
    if self.Validator in {validators.Draft3Validator, validators.Draft4Validator}:
        with self.assertRaises(exceptions.SchemaError):
            self.Validator.check_schema({'enum': []})
    else:
        self.Validator.check_schema({'enum': []})