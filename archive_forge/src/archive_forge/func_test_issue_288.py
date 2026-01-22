from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_288(self):
    import sys
    from srsly.ruamel_yaml.compat import StringIO
    from srsly.ruamel_yaml import YAML
    yamldoc = dedent('        ---\n        # Reusable values\n        aliases:\n          # First-element comment\n          - &firstEntry First entry\n          # Second-element comment\n          - &secondEntry Second entry\n\n          # Third-element comment is\n          # a multi-line value\n          - &thirdEntry Third entry\n\n        # EOF Comment\n        ')
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.explicit_start = True
    yaml.preserve_quotes = True
    yaml.width = sys.maxsize
    data = yaml.load(yamldoc)
    buf = StringIO()
    yaml.dump(data, buf)
    assert buf.getvalue() == yamldoc