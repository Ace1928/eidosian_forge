import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_concise_help_text(self):
    """Concise help text excludes the descriptive sections."""

    class cmd_Demo(commands.Command):
        __doc__ = 'A sample command.\n\n            Blah blah blah.\n\n            :Examples:\n                Example 1::\n\n                    cmd arg1\n            '
    cmd = cmd_Demo()
    helptext = cmd.get_help_text()
    self.assertEqualDiff('Purpose: A sample command.\nUsage:   brz Demo\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nDescription:\n  Blah blah blah.\n\nExamples:\n    Example 1:\n\n        cmd arg1\n\n', helptext)
    helptext = cmd.get_help_text(verbose=False)
    self.assertEqualDiff('Purpose: A sample command.\nUsage:   brz Demo\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nSee brz help Demo for more details and examples.\n\n', helptext)