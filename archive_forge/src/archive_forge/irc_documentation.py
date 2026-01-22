import sys
from pygments.formatter import Formatter
from pygments.token import Keyword, Name, Comment, String, Error, \
from pygments.util import get_choice_opt

    Format tokens with IRC color sequences

    The `get_style_defs()` method doesn't do anything special since there is
    no support for common styles.

    Options accepted:

    `bg`
        Set to ``"light"`` or ``"dark"`` depending on the terminal's background
        (default: ``"light"``).

    `colorscheme`
        A dictionary mapping token types to (lightbg, darkbg) color names or
        ``None`` (default: ``None`` = use builtin colorscheme).

    `linenos`
        Set to ``True`` to have line numbers in the output as well
        (default: ``False`` = no line numbers).
    