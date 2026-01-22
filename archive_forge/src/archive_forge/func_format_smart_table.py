import collections
import re
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import (
def format_smart_table(data, column_names):
    """
    Render tabular data using the most appropriate representation.

    :param data: An iterable (e.g. a :func:`tuple` or :class:`list`)
                 containing the rows of the table, where each row is an
                 iterable containing the columns of the table (strings).
    :param column_names: An iterable of column names (strings).
    :returns: The rendered table (a string).

    If you want an easy way to render tabular data on a terminal in a human
    friendly format then this function is for you! It works as follows:

    - If the input data doesn't contain any line breaks the function
      :func:`format_pretty_table()` is used to render a pretty table. If the
      resulting table fits in the terminal without wrapping the rendered pretty
      table is returned.

    - If the input data does contain line breaks or if a pretty table would
      wrap (given the width of the terminal) then the function
      :func:`format_robust_table()` is used to render a more robust table that
      can deal with data containing line breaks and long text.
    """
    data = [normalize_columns(r) for r in data]
    column_names = normalize_columns(column_names)
    if not any((any(('\n' in c for c in r)) for r in data)):
        pretty_table = format_pretty_table(data, column_names)
        table_width = max(map(ansi_width, pretty_table.splitlines()))
        num_rows, num_columns = find_terminal_size()
        if table_width <= num_columns:
            return pretty_table
    return format_robust_table(data, column_names)