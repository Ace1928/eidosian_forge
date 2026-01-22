from statsmodels.compat.python import lmap, lrange
from itertools import cycle, zip_longest
import csv
def get_colwidths(self, output_format, **fmt_dict):
    """Return list, the widths of each column."""
    call_args = [output_format]
    for k, v in sorted(fmt_dict.items()):
        if isinstance(v, list):
            call_args.append((k, tuple(v)))
        elif isinstance(v, dict):
            call_args.append((k, tuple(sorted(v.items()))))
        else:
            call_args.append((k, v))
    key = tuple(call_args)
    try:
        return self._colwidths[key]
    except KeyError:
        self._colwidths[key] = self._get_colwidths(output_format, **fmt_dict)
        return self._colwidths[key]