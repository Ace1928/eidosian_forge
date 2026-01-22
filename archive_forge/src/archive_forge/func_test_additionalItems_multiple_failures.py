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
def test_additionalItems_multiple_failures(self):
    message = self.message_for(instance=[1, 2, 3], schema={'items': [], 'additionalItems': False}, cls=validators.Draft3Validator)
    self.assertIn('(1, 2, 3 were unexpected)', message)