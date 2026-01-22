from __future__ import absolute_import, print_function
from functools import partial
import re
from .compat import text_type, binary_type
def load_yaml_guess_indent(stream, **kw):
    """guess the indent and block sequence indent of yaml stream/string

    returns round_trip_loaded stream, indent level, block sequence indent
    - block sequence indent is the number of spaces before a dash relative to previous indent
    - if there are no block sequences, indent is taken from nested mappings, block sequence
      indent is unset (None) in that case
    """
    from .main import round_trip_load

    def leading_spaces(l):
        idx = 0
        while idx < len(l) and l[idx] == ' ':
            idx += 1
        return idx
    if isinstance(stream, text_type):
        yaml_str = stream
    elif isinstance(stream, binary_type):
        yaml_str = stream.decode('utf-8')
    else:
        yaml_str = stream.read()
    map_indent = None
    indent = None
    block_seq_indent = None
    prev_line_key_only = None
    key_indent = 0
    for line in yaml_str.splitlines():
        rline = line.rstrip()
        lline = rline.lstrip()
        if lline.startswith('- '):
            l_s = leading_spaces(line)
            block_seq_indent = l_s - key_indent
            idx = l_s + 1
            while line[idx] == ' ':
                idx += 1
            if line[idx] == '#':
                continue
            indent = idx - key_indent
            break
        if map_indent is None and prev_line_key_only is not None and rline:
            idx = 0
            while line[idx] in ' -':
                idx += 1
            if idx > prev_line_key_only:
                map_indent = idx - prev_line_key_only
        if rline.endswith(':'):
            key_indent = leading_spaces(line)
            idx = 0
            while line[idx] == ' ':
                idx += 1
            prev_line_key_only = idx
            continue
        prev_line_key_only = None
    if indent is None and map_indent is not None:
        indent = map_indent
    return (round_trip_load(yaml_str, **kw), indent, block_seq_indent)