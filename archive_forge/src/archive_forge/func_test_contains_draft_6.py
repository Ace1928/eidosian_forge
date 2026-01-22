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
def test_contains_draft_6(self):
    schema = {'contains': {'const': 12}}
    message = self.message_for(instance=[2, {}, []], schema=schema, cls=validators.Draft6Validator)
    self.assertEqual(message, 'None of [2, {}, []] are valid under the given schema')