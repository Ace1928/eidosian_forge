import argparse
import io
from unittest import mock
from cliff.formatters import shell
from cliff.tests import base
from cliff.tests import test_columns
def test_non_bash_friendly_values(self):
    sf = shell.ShellFormatter()
    c = ('a', 'foo-bar', 'provider:network_type')
    d = (True, 'baz', 'vxlan')
    expected = 'a="True"\nfoo_bar="baz"\nprovider_network_type="vxlan"\n'
    output = io.StringIO()
    args = mock.Mock()
    args.variables = ['a', 'foo-bar', 'provider:network_type']
    args.prefix = ''
    sf.emit_one(c, d, output, args)
    actual = output.getvalue()
    self.assertEqual(expected, actual)