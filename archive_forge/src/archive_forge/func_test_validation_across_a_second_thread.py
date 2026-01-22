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
def test_validation_across_a_second_thread(self):
    failed = []

    def validate():
        try:
            validators.validate(instance=37, schema=True)
        except:
            failed.append(sys.exc_info())
    validate()
    from threading import Thread
    thread = Thread(target=validate)
    thread.start()
    thread.join()
    self.assertEqual((thread.is_alive(), failed), (False, []))