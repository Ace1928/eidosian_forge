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
def test_draft_201909(self):
    schema = {'$schema': 'https://json-schema.org/draft/2019-09/schema'}
    self.assertIs(validators.validator_for(schema), validators.Draft201909Validator)
    schema = {'$schema': 'https://json-schema.org/draft/2019-09/schema#'}
    self.assertIs(validators.validator_for(schema), validators.Draft201909Validator)