from __future__ import unicode_literals
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from six.moves import range
def scroll_one_line_down(event):
    """
    scroll_offset += 1
    """
    w = find_window_for_buffer_name(event.cli, event.cli.current_buffer_name)
    b = event.cli.current_buffer
    if w:
        if w.render_info:
            info = w.render_info
            if w.vertical_scroll < info.content_height - info.window_height:
                if info.cursor_position.y <= info.configured_scroll_offsets.top:
                    b.cursor_position += b.document.get_cursor_down_position()
                w.vertical_scroll += 1