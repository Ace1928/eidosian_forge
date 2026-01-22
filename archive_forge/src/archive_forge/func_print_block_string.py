from typing import Collection, List
from sys import maxsize
def print_block_string(value: str, minimize: bool=False) -> str:
    """Print a block string in the indented block form.

    Prints a block string in the indented block form by adding a leading and
    trailing blank line. However, if a block string starts with whitespace and
    is a single-line, adding a leading blank line would strip that whitespace.

    For internal use only.
    """
    if not isinstance(value, str):
        value = str(value)
    escaped_value = value.replace('"""', '\\"""')
    lines = escaped_value.splitlines() or ['']
    num_lines = len(lines)
    is_single_line = num_lines == 1
    force_leading_new_line = num_lines > 1 and all((not line or line[0] in ' \t' for line in lines[1:]))
    has_trailing_triple_quotes = escaped_value.endswith('\\"""')
    has_trailing_quote = value.endswith('"') and (not has_trailing_triple_quotes)
    has_trailing_slash = value.endswith('\\')
    force_trailing_new_line = has_trailing_quote or has_trailing_slash
    print_as_multiple_lines = not minimize and (not is_single_line or len(value) > 70 or force_trailing_new_line or force_leading_new_line or has_trailing_triple_quotes)
    skip_leading_new_line = is_single_line and value and (value[0] in ' \t')
    before = '\n' if print_as_multiple_lines and (not skip_leading_new_line) or force_leading_new_line else ''
    after = '\n' if print_as_multiple_lines or force_trailing_new_line else ''
    return f'"""{before}{escaped_value}{after}"""'