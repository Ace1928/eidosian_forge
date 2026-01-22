import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_anchor_id_retained(self):
    data = load('\n        a: &id002\n          b: 1\n          c: 2\n        d: *id002\n        e: &etemplate\n          b: 1\n          c: 2\n        f: *etemplate\n        ')
    compare(data, '\n        a: &id001\n          b: 1\n          c: 2\n        d: *id001\n        e: &etemplate\n          b: 1\n          c: 2\n        f: *etemplate\n        ')