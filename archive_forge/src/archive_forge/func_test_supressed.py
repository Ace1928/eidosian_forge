import argparse
import textwrap
from cliff import sphinxext
from cliff.tests import base
def test_supressed(self):
    """Handle a supressed action."""
    parser = argparse.ArgumentParser(prog='hello-world', add_help=False)
    parser.add_argument('name', help='user name')
    parser.add_argument('--variable', help=argparse.SUPPRESS)
    output = '\n'.join(sphinxext._format_parser(parser))
    self.assertEqual(textwrap.dedent('\n        .. program:: hello-world\n        .. code-block:: shell\n\n            hello-world name\n\n\n        .. option:: name\n\n            user name\n        ').lstrip(), output)