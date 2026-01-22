from __future__ import unicode_literals
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Point, Screen, WritePosition
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.token import Token
from prompt_toolkit.utils import is_windows
from six.moves import range
def output_char(char):
    """
        Write the output of this character.
        """
    the_last_token = last_token[0]
    if the_last_token and the_last_token == char.token:
        write(char.char)
    else:
        _output_set_attributes(attrs_for_token[char.token])
        write(char.char)
        last_token[0] = char.token