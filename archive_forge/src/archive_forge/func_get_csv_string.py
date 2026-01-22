from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def get_csv_string(self, **kwargs) -> str:
    """Return string representation of CSV formatted table in the current state

        Keyword arguments are first interpreted as table formatting options, and
        then any unused keyword arguments are passed to csv.writer(). For
        example, get_csv_string(header=False, delimiter='	') would use
        header as a PrettyTable formatting option (skip the header row) and
        delimiter as a csv.writer keyword argument.
        """
    import csv
    options = self._get_options(kwargs)
    csv_options = {key: value for key, value in kwargs.items() if key not in options}
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer, **csv_options)
    if options.get('header'):
        csv_writer.writerow(self._field_names)
    for row in self._get_rows(options):
        csv_writer.writerow(row)
    return csv_buffer.getvalue()