import argparse
import textwrap
from cliff import sphinxext
from cliff.tests import base
def test_empty_help(self):
    """Handle positional and optional actions without help messages."""
    parser = argparse.ArgumentParser(prog='hello-world', add_help=False)
    parser.add_argument('name', action='store')
    parser.add_argument('--language', dest='lang')
    output = '\n'.join(sphinxext._format_parser(parser))
    self.assertEqual(textwrap.dedent('\n        .. program:: hello-world\n        .. code-block:: shell\n\n            hello-world [--language LANG] name\n\n        .. option:: --language <LANG>\n\n        .. option:: name\n        ').lstrip(), output)