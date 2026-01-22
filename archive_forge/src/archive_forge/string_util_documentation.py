from __future__ import annotations
import re
import textwrap
from typing import TYPE_CHECKING, Any, Final, cast
from streamlit.errors import StreamlitAPIException
Returns True if the given string contains what seem to be HTML tags.

    Note that false positives/negatives are possible, so this function should not be
    used in contexts where complete correctness is required.