from __future__ import unicode_literals
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from six.moves import range
def scroll_one_line_up(event):
    """
    scroll_offset -= 1
    """
    w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
    b = event.cli.current_buffer
    if w:
        if w.render_info:
            info = w.render_info
            if w.vertical_scroll > 0:
                first_line_height = info.get_height_for_line(info.first_visible_line())
                cursor_up = info.cursor_position.y - (info.window_height - 1 - first_line_height - info.configured_scroll_offsets.bottom)
                for _ in range(max(0, cursor_up)):
                    b.cursor_position += b.document.get_cursor_up_position()
                w.vertical_scroll -= 1