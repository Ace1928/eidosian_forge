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
def test_custom_validator_draft6(self):
    Validator = validators.create(meta_schema={'$id': 'meta schema $id'}, version='13')
    schema = {'$schema': 'meta schema $id'}
    self.assertIs(validators.validator_for(schema), Validator)