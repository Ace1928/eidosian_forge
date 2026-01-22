import re
def expand_abbreviations(self, text: str) -> str:
    """
        Expands the abbreviate words.
        """
    for regex, replacement in self._abbreviations:
        text = re.sub(regex, replacement, text)
    return text