import sys, os
import textwrap
def store_option_strings(self, parser):
    self.indent()
    max_len = 0
    for opt in parser.option_list:
        strings = self.format_option_strings(opt)
        self.option_strings[opt] = strings
        max_len = max(max_len, len(strings) + self.current_indent)
    self.indent()
    for group in parser.option_groups:
        for opt in group.option_list:
            strings = self.format_option_strings(opt)
            self.option_strings[opt] = strings
            max_len = max(max_len, len(strings) + self.current_indent)
    self.dedent()
    self.dedent()
    self.help_position = min(max_len + 2, self.max_help_position)
    self.help_width = max(self.width - self.help_position, 11)