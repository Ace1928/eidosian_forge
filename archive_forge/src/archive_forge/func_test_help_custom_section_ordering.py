import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_help_custom_section_ordering(self):
    """Custom descriptive sections should remain in the order given."""

    class cmd_Demo(commands.Command):
        __doc__ = 'A sample command.\n\n            Blah blah blah.\n\n            :Formats:\n              Interesting stuff about formats.\n\n            :Examples:\n              Example 1::\n\n                cmd arg1\n\n            :Tips:\n              Clever things to keep in mind.\n            '
    self.assertCmdHelp('zz{{:Purpose: zz{{A sample command.}}\n}}zz{{:Usage:   brz Demo\n}}\nzz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nDescription:\n  zz{{zz{{Blah blah blah.}}\n\n}}:Formats:\n  zz{{Interesting stuff about formats.}}\n\nExamples:\n  zz{{Example 1::}}\n\n    zz{{cmd arg1}}\n\nTips:\n  zz{{Clever things to keep in mind.}}\n\n', cmd_Demo())