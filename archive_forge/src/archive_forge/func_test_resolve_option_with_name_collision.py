import argparse
import functools
from cliff import command
from cliff.tests import base
def test_resolve_option_with_name_collision(self):
    cmd = TestCommand(None, None)
    parser = cmd.get_parser('NAME')
    parser.add_argument('-z', '--zero', dest='zero', default='zero-default')
    args = parser.parse_args(['-z', 'foo', 'a', 'b'])
    self.assertEqual(args.zippy, 'foo')
    self.assertEqual(args.zero, 'zero-default')