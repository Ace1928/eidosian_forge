import warnings
from pathlib import Path
from typing import Optional
from .. import constants
from ._token import get_token
@classmethod
def save_token(cls, token: str) -> None:
    """
        Save token, creating folder as needed.

        Token is saved in the huggingface home folder. You can configure it by setting
        the `HF_HOME` environment variable.

        Args:
            token (`str`):
                The token to save to the [`HfFolder`]
        """
    cls.path_token.parent.mkdir(parents=True, exist_ok=True)
    cls.path_token.write_text(token)