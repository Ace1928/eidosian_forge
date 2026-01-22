import collections
import re
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import (
def format_pretty_table(data, column_names=None, horizontal_bar='-', vertical_bar='|'):
    """
    Render a table using characters like dashes and vertical bars to emulate borders.

    :param data: An iterable (e.g. a :func:`tuple` or :class:`list`)
                 containing the rows of the table, where each row is an
                 iterable containing the columns of the table (strings).
    :param column_names: An iterable of column names (strings).
    :param horizontal_bar: The character used to represent a horizontal bar (a
                           string).
    :param vertical_bar: The character used to represent a vertical bar (a
                         string).
    :returns: The rendered table (a string).

    Here's an example:

    >>> from humanfriendly.tables import format_pretty_table
    >>> column_names = ['Version', 'Uploaded on', 'Downloads']
    >>> humanfriendly_releases = [
    ... ['1.23', '2015-05-25', '218'],
    ... ['1.23.1', '2015-05-26', '1354'],
    ... ['1.24', '2015-05-26', '223'],
    ... ['1.25', '2015-05-26', '4319'],
    ... ['1.25.1', '2015-06-02', '197'],
    ... ]
    >>> print(format_pretty_table(humanfriendly_releases, column_names))
    -------------------------------------
    | Version | Uploaded on | Downloads |
    -------------------------------------
    | 1.23    | 2015-05-25  |       218 |
    | 1.23.1  | 2015-05-26  |      1354 |
    | 1.24    | 2015-05-26  |       223 |
    | 1.25    | 2015-05-26  |      4319 |
    | 1.25.1  | 2015-06-02  |       197 |
    -------------------------------------

    Notes about the resulting table:

    - If a column contains numeric data (integer and/or floating point
      numbers) in all rows (ignoring column names of course) then the content
      of that column is right-aligned, as can be seen in the example above. The
      idea here is to make it easier to compare the numbers in different
      columns to each other.

    - The column names are highlighted in color so they stand out a bit more
      (see also :data:`.HIGHLIGHT_COLOR`). The following screen shot shows what
      that looks like (my terminals are always set to white text on a black
      background):

      .. image:: images/pretty-table.png
    """
    data = [normalize_columns(r, expandtabs=True) for r in data]
    if column_names is not None:
        column_names = normalize_columns(column_names)
        if column_names:
            if terminal_supports_colors():
                column_names = [highlight_column_name(n) for n in column_names]
            data.insert(0, column_names)
    widths = collections.defaultdict(int)
    numeric_data = collections.defaultdict(list)
    for row_index, row in enumerate(data):
        for column_index, column in enumerate(row):
            widths[column_index] = max(widths[column_index], ansi_width(column))
            if not (column_names and row_index == 0):
                numeric_data[column_index].append(bool(NUMERIC_DATA_PATTERN.match(ansi_strip(column))))
    line_delimiter = horizontal_bar * (sum(widths.values()) + len(widths) * 3 + 1)
    lines = [line_delimiter]
    for row_index, row in enumerate(data):
        line = [vertical_bar]
        for column_index, column in enumerate(row):
            padding = ' ' * (widths[column_index] - ansi_width(column))
            if all(numeric_data[column_index]):
                line.append(' ' + padding + column + ' ')
            else:
                line.append(' ' + column + padding + ' ')
            line.append(vertical_bar)
        lines.append(u''.join(line))
        if column_names and row_index == 0:
            lines.append(line_delimiter)
    lines.append(line_delimiter)
    return u'\n'.join(lines)