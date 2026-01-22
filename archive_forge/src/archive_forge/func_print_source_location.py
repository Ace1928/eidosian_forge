import re
from typing import Optional, Tuple, cast
from .ast import Location
from .location import SourceLocation, get_location
from .source import Source
def print_source_location(source: Source, source_location: SourceLocation) -> str:
    """Render a helpful description of the location in the GraphQL Source document."""
    first_line_column_offset = source.location_offset.column - 1
    body = ''.rjust(first_line_column_offset) + source.body
    line_index = source_location.line - 1
    line_offset = source.location_offset.line - 1
    line_num = source_location.line + line_offset
    column_offset = first_line_column_offset if source_location.line == 1 else 0
    column_num = source_location.column + column_offset
    location_str = f'{source.name}:{line_num}:{column_num}\n'
    lines = _re_newline.split(body)
    location_line = lines[line_index]
    if len(location_line) > 120:
        sub_line_index, sub_line_column_num = divmod(column_num, 80)
        sub_lines = [location_line[i:i + 80] for i in range(0, len(location_line), 80)]
        return location_str + print_prefixed_lines((f'{line_num} |', sub_lines[0]), *[('|', sub_line) for sub_line in sub_lines[1:sub_line_index + 1]], ('|', '^'.rjust(sub_line_column_num)), ('|', sub_lines[sub_line_index + 1] if sub_line_index < len(sub_lines) - 1 else None))
    return location_str + print_prefixed_lines((f'{line_num - 1} |', lines[line_index - 1] if line_index > 0 else None), (f'{line_num} |', location_line), ('|', '^'.rjust(column_num)), (f'{line_num + 1} |', lines[line_index + 1] if line_index < len(lines) - 1 else None))