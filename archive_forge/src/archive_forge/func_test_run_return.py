import argparse
import functools
from cliff import command
from cliff.tests import base
def test_run_return(self):
    cmd = TestCommand(None, None, cmd_name='object action')
    assert cmd.run(None) == 42