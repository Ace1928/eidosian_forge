from __future__ import annotations
import json
import pickle  # use pickle, not cPickle so that we get the traceback in case of errors
import string
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from unittest import TestCase
import pytest
from monty.json import MontyDecoder, MontyEncoder, MSONable
from monty.serialization import loadfn
from pymatgen.core import ROOT, SETTINGS, Structure
@staticmethod
def assert_str_content_equal(actual, expected):
    """Tests if two strings are equal, ignoring things like trailing spaces, etc."""
    strip_whitespace = {ord(c): None for c in string.whitespace}
    return actual.translate(strip_whitespace) == expected.translate(strip_whitespace)