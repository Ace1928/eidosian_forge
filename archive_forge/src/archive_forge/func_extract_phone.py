import re
import html
import contextlib
from typing import Optional
def extract_phone(text: str) -> Optional[str]:
    """
    Extracts the phone
    """
    match = re.search('[\\d\\(\\)\\-]+', text)
    if not match:
        return None
    pn = match[0]
    if len(pn.replace('(', '').replace(')', '').replace('-', '').replace(' ', '')) <= 9:
        return None
    return pn