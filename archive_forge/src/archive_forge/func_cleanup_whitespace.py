import re
import html
import contextlib
from typing import Optional
def cleanup_whitespace(text: str) -> str:
    """
    Cleans up the whitespace
    """
    return re.sub('\\s+', ' ', text).strip()