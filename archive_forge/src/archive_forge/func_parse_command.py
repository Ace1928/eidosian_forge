import argparse
import ast
import re
import sys
def parse_command(command):
    """Parse command string into a list of arguments.

  - Disregards whitespace inside double quotes and brackets.
  - Strips paired leading and trailing double quotes in arguments.
  - Splits the command at whitespace.

  Nested double quotes and brackets are not handled.

  Args:
    command: (str) Input command.

  Returns:
    (list of str) List of arguments.
  """
    command = command.strip()
    if not command:
        return []
    brackets_intervals = [f.span() for f in _BRACKETS_PATTERN.finditer(command)]
    quotes_intervals = [f.span() for f in _QUOTES_PATTERN.finditer(command)]
    whitespaces_intervals = [f.span() for f in _WHITESPACE_PATTERN.finditer(command)]
    if not whitespaces_intervals:
        return [command]
    arguments = []
    idx0 = 0
    for start, end in whitespaces_intervals + [(len(command), None)]:
        if not any((interval[0] < start < interval[1] for interval in brackets_intervals + quotes_intervals)):
            argument = command[idx0:start]
            if argument.startswith('"') and argument.endswith('"') or (argument.startswith("'") and argument.endswith("'")):
                argument = argument[1:-1]
            arguments.append(argument)
            idx0 = end
    return arguments