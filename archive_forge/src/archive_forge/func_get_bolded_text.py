from typing import Dict, List, Optional, TextIO
def get_bolded_text(text: str) -> str:
    """Get bolded text."""
    return f'\x1b[1m{text}\x1b[0m'