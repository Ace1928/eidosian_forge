from __future__ import unicode_literals
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from six.moves import range
def scroll_forward(event, half=False):
    """
    Scroll window down.
    """
    w = _current_window_for_event(event)
    b = event.cli.current_buffer
    if w and w.render_info:
        info = w.render_info
        ui_content = info.ui_content
        scroll_height = info.window_height
        if half:
            scroll_height //= 2
        y = b.document.cursor_position_row + 1
        height = 0
        while y < ui_content.line_count:
            line_height = info.get_height_for_line(y)
            if height + line_height < scroll_height:
                height += line_height
                y += 1
            else:
                break
        b.cursor_position = b.document.translate_row_col_to_index(y, 0)