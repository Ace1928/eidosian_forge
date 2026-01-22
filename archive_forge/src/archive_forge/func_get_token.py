import os
import warnings
from pathlib import Path
from threading import Lock
from typing import Optional
from .. import constants
from ._runtime import is_google_colab
def get_token() -> Optional[str]:
    """
    Get token if user is logged in.

    Note: in most cases, you should use [`huggingface_hub.utils.build_hf_headers`] instead. This method is only useful
          if you want to retrieve the token for other purposes than sending an HTTP request.

    Token is retrieved in priority from the `HF_TOKEN` environment variable. Otherwise, we read the token file located
    in the Hugging Face home folder. Returns None if user is not logged in. To log in, use [`login`] or
    `huggingface-cli login`.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.
    """
    return _get_token_from_google_colab() or _get_token_from_environment() or _get_token_from_file()