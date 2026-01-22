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
def test_check_schema_with_different_metaschema_defaults_to_self(self):
    """
        A validator whose metaschema doesn't declare $schema defaults to its
        own validation behavior, not the latest "normal" specification.
        """
    NoEmptySchemasValidator = validators.create(meta_schema={'fail': [{'message': 'Meta schema whoops!'}]}, validators={'fail': fail})
    with self.assertRaises(exceptions.SchemaError):
        NoEmptySchemasValidator.check_schema({})