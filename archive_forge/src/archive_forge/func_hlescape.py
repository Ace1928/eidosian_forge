import re
from typing import Dict, Optional
def hlescape(s: str, latex_engine: Optional[str]=None) -> str:
    """Escape text for LaTeX highlighter."""
    if latex_engine in ('lualatex', 'xelatex'):
        return s.translate(_tex_hlescape_map_without_unicode)
    else:
        return s.translate(_tex_hlescape_map)