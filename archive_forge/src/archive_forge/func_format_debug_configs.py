from OpenGL.EGL import *
import itertools
def format_debug_configs(debug_configs, formats=CONFIG_FORMAT):
    """Format config for compact debugging display
    
    Produces a config summary display for a set of 
    debug_configs as a text-mode table.

    Uses `formats` (default `CONFIG_FORMAT`) to determine 
    which fields are extracted and how they are formatted
    along with the column/subcolum set to be rendered in
    the overall header.

    returns formatted ASCII table for display in debug
    logs or utilities
    """
    columns = []
    for key, format, subcol, col in formats:
        column = []
        max_width = 0
        for row in debug_configs:
            if isinstance(row, EGLConfig):
                raise TypeError(row, 'Call debug_config(display,config)')
            try:
                value = row[key.name]
            except KeyError:
                formatted = '_'
            else:
                if isinstance(format, str):
                    formatted = format % value
                else:
                    formatted = format(value)
            max_width = max((len(formatted), max_width))
            column.append(formatted)
        columns.append({'rows': column, 'key': key, 'format': format, 'subcol': subcol, 'col': col, 'width': max_width})
    headers = []
    subheaders = []
    rows = [headers, subheaders]
    last_column = None
    last_column_width = 0
    for header, subcols in itertools.groupby(columns, lambda x: x['col']):
        subcols = list(subcols)
        width = sum([col['width'] for col in subcols]) + (len(subcols) - 1)
        headers.append(header.center(width, '.')[:width])
    for column in columns:
        subheaders.append(column['subcol'].rjust(column['width'])[:column['width']])
    rows.extend(zip(*[[v.rjust(col['width'], ' ') for v in col['rows']] for col in columns]))
    return '\n'.join([' '.join(row) for row in rows])