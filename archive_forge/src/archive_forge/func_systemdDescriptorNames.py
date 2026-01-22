from hypothesis.strategies import SearchStrategy, characters, text
from typing_extensions import Literal
def systemdDescriptorNames() -> SearchStrategy[str]:
    """
    Build strings that are legal values for the systemd
    I{FileDescriptorName} field.
    """
    control_characters: Literal['Cc'] = 'Cc'
    return text(min_size=1, max_size=255, alphabet=characters(min_codepoint=0, max_codepoint=127, blacklist_categories=(control_characters,), blacklist_characters=(':',)))