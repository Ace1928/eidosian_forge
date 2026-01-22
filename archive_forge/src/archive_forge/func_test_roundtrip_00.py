import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_roundtrip_00(self):
    yaml_str = '\n        - &dotted.words.here\n          a: 1\n          b: 2\n        - *dotted.words.here\n        '
    data = round_trip(yaml_str)