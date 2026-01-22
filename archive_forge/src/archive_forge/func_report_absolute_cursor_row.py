from __future__ import unicode_literals
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Point, Screen, WritePosition
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.token import Token
from prompt_toolkit.utils import is_windows
from six.moves import range
def report_absolute_cursor_row(self, row):
    """
        To be called when we know the absolute cursor position.
        (As an answer of a "Cursor Position Request" response.)
        """
    total_rows = self.output.get_size().rows
    rows_below_cursor = total_rows - row + 1
    self._min_available_height = rows_below_cursor
    self.waiting_for_cpr = False