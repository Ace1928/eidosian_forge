import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_formatted_help_text(self):
    """Help text should be plain text by default."""

    class cmd_Demo(commands.Command):
        __doc__ = 'A sample command.\n\n            :Examples:\n                Example 1::\n\n                    cmd arg1\n\n                Example 2::\n\n                    cmd arg2\n\n                A code block follows.\n\n                ::\n\n                    brz Demo something\n            '
    cmd = cmd_Demo()
    helptext = cmd.get_help_text()
    self.assertEqualDiff('Purpose: A sample command.\nUsage:   brz Demo\n\nOptions:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\nExamples:\n    Example 1:\n\n        cmd arg1\n\n    Example 2:\n\n        cmd arg2\n\n    A code block follows.\n\n        brz Demo something\n\n', helptext)
    helptext = cmd.get_help_text(plain=False)
    self.assertEqualDiff(':Purpose: A sample command.\n:Usage:   brz Demo\n\n:Options:\n  -h, --help     Show help message.\n  -q, --quiet    Only display errors and warnings.\n  --usage        Show usage message and options.\n  -v, --verbose  Display more information.\n\n:Examples:\n    Example 1::\n\n        cmd arg1\n\n    Example 2::\n\n        cmd arg2\n\n    A code block follows.\n\n    ::\n\n        brz Demo something\n\n', helptext)