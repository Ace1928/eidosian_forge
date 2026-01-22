import sys
import textwrap
from pathlib import Path
def na_round_trip(inp, outp=None, extra=None, intermediate=None, indent=None, top_level_colon_align=None, prefix_colon=None, preserve_quotes=None, explicit_start=None, explicit_end=None, version=None, dump_data=None):
    """
    inp:    input string to parse
    outp:   expected output (equals input if not specified)
    """
    inp = dedent(inp)
    if outp is None:
        outp = inp
    if version is not None:
        version = version
    doutp = dedent(outp)
    if extra is not None:
        doutp += extra
    yaml = YAML()
    yaml.preserve_quotes = preserve_quotes
    yaml.scalar_after_indicator = False
    data = yaml.load(inp)
    if dump_data:
        print('data', data)
    if intermediate is not None:
        if isinstance(intermediate, dict):
            for k, v in intermediate.items():
                if data[k] != v:
                    print('{0!r} <> {1!r}'.format(data[k], v))
                    raise ValueError
    yaml.indent = indent
    yaml.top_level_colon_align = top_level_colon_align
    yaml.prefix_colon = prefix_colon
    yaml.explicit_start = explicit_start
    yaml.explicit_end = explicit_end
    res = yaml.dump(data, compare=doutp)
    return res