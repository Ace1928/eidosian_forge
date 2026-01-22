import argparse
import functools
from cliff import command
from cliff.tests import base
def test_option_name_collision_with_alias(self):
    cmd = TestCommand(None, None)
    parser = cmd.get_parser('NAME')
    parser.add_argument('-z', '--zero')