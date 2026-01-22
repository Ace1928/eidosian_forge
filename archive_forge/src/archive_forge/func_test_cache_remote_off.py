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
def test_cache_remote_off(self):
    response = [object()]

    def handler(url):
        try:
            return response.pop()
        except IndexError:
            self.fail('Handler called twice!')
    ref = 'foo://bar'
    resolver = validators._RefResolver('', {}, cache_remote=False, handlers={'foo': handler})
    with resolver.resolving(ref):
        pass