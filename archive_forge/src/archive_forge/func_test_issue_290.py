from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_290(self):
    import sys
    from srsly.ruamel_yaml.compat import StringIO
    from srsly.ruamel_yaml import YAML
    yamldoc = dedent('        ---\n        aliases:\n          # Folded-element comment\n          # for a multi-line value\n          - &FoldedEntry >\n            THIS IS A\n            FOLDED, MULTI-LINE\n            VALUE\n\n          # Literal-element comment\n          # for a multi-line value\n          - &literalEntry |\n            THIS IS A\n            LITERAL, MULTI-LINE\n            VALUE\n\n          # Plain-element comment\n          - &plainEntry Plain entry\n        ')
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.explicit_start = True
    yaml.preserve_quotes = True
    yaml.width = sys.maxsize
    data = yaml.load(yamldoc)
    buf = StringIO()
    yaml.dump(data, buf)
    assert buf.getvalue() == yamldoc