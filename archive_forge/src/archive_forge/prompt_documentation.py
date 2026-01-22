from __future__ import unicode_literals
from six import text_type
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER
from prompt_toolkit.token import Token
from .utils import token_list_len
from .processors import Processor, Transformation

        Create a default prompt with a static message text.
        