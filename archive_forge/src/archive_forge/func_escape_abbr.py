import re
from typing import Dict, Optional
def escape_abbr(text: str) -> str:
    """Adjust spacing after abbreviations. Works with @ letter or other."""
    return re.sub('\\.(?=\\s|$)', '.\\@{}', text)