from typing import List
from typing import Optional
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.nodes import Item
Truncate given list of strings that makes up the assertion explanation.

    Truncates to either 8 lines, or 640 characters - whichever the input reaches
    first, taking the truncation explanation into account. The remaining lines
    will be replaced by a usage message.
    