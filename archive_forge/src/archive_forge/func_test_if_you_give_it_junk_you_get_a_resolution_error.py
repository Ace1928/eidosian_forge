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
def test_if_you_give_it_junk_you_get_a_resolution_error(self):
    error = ValueError("Oh no! What's this?")

    def handler(url):
        raise error
    ref = 'foo://bar'
    resolver = validators._RefResolver('', {}, handlers={'foo': handler})
    with self.assertRaises(exceptions._RefResolutionError) as err:
        with resolver.resolving(ref):
            self.fail("Shouldn't get this far!")
    self.assertEqual(err.exception, exceptions._RefResolutionError(error))