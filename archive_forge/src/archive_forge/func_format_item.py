from __future__ import unicode_literals
import math
import textwrap
import re
from cmakelang import common
def format_item(config, line_width, item):
    """
  Return lines of formatted text based on the typeof markup
  """
    if item.kind == CommentType.SEPARATOR:
        return ['']
    if item.kind == CommentType.FENCE:
        return ['~~~']
    if item.kind == CommentType.VERBATIM:
        return [line.rstrip() for line in item.lines]
    if is_hashruler(item) and config.markup.canonicalize_hashrulers:
        return ['#' * line_width]
    if item.kind in (CommentType.PARAGRAPH, CommentType.NOTE, CommentType.RULER):
        wrapper = textwrap.TextWrapper(width=line_width, **COMMON_KWARGS)
        return common.stable_wrap(wrapper, '\n'.join(item.lines).strip())
    if item.kind == CommentType.BULLET_LIST:
        assert line_width > 2
        outlines = []
        wrapper = textwrap.TextWrapper(width=line_width - 2, **COMMON_KWARGS)
        for line in item.lines:
            increment_lines = common.stable_wrap(wrapper, line.strip())
            outlines.append(config.markup.bullet_char + ' ' + increment_lines[0])
            outlines.extend(('  ' + iline for iline in increment_lines[1:]))
        return outlines
    if item.kind == CommentType.ENUM_LIST:
        assert line_width > 2
        outlines = []
        wrapper = textwrap.TextWrapper(width=line_width - 2, **COMMON_KWARGS)
        digits = int(math.ceil(math.log(len(item.lines), 10)))
        fmt = '{:%dd}%s ' % (digits, config.markup.enum_char)
        indent = ' ' * (digits + 2)
        for idx, line in enumerate(item.lines):
            increment_lines = common.stable_wrap(wrapper, line.strip())
            outlines.append(fmt.format(idx + 1) + increment_lines[0])
            outlines.extend((indent + iline for iline in increment_lines[1:]))
        return outlines
    raise AssertionError('Unexepected case')