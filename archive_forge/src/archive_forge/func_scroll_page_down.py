from __future__ import unicode_literals
from prompt_toolkit.layout.utils import find_window_for_buffer_name
from six.moves import range
def scroll_page_down(event):
    """
    Scroll page down. (Prefer the cursor at the top of the page, after scrolling.)
    """
    w = _current_window_for_event(event)
    b = event.cli.current_buffer
    if w and w.render_info:
        line_index = max(w.render_info.last_visible_line(), w.vertical_scroll + 1)
        w.vertical_scroll = line_index
        b.cursor_position = b.document.translate_row_col_to_index(line_index, 0)
        b.cursor_position += b.document.get_start_of_line_position(after_whitespace=True)