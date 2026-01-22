import argparse
import functools
from cliff import command
from cliff.tests import base
def test_get_description_default(self):
    cmd = TestCommandNoDocstring(None, None)
    desc = cmd.get_description()
    assert desc == ''