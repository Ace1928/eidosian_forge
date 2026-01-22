import sys, os
import textwrap
def expand_default(self, option):
    if self.parser is None or not self.default_tag:
        return option.help
    default_value = self.parser.defaults.get(option.dest)
    if default_value is NO_DEFAULT or default_value is None:
        default_value = self.NO_DEFAULT_VALUE
    return option.help.replace(self.default_tag, str(default_value))