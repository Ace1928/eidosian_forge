import os
import subprocess
from functools import partial
from getpass import getpass
from pathlib import Path
from typing import Optional
from . import constants
from .commands._cli_utils import ANSI
from .utils import (
from .utils._token import _get_token_from_environment, _get_token_from_google_colab
def logout() -> None:
    """Logout the machine from the Hub.

    Token is deleted from the machine and removed from git credential.
    """
    if get_token() is None:
        print('Not logged in!')
        return
    unset_git_credential()
    try:
        Path(constants.HF_TOKEN_PATH).unlink()
    except FileNotFoundError:
        pass
    if _get_token_from_google_colab() is not None:
        raise EnvironmentError('You are automatically logged in using a Google Colab secret.\nTo log out, you must unset the `HF_TOKEN` secret in your Colab settings.')
    if _get_token_from_environment() is not None:
        raise EnvironmentError('Token has been deleted from your machine but you are still logged in.\nTo log out, you must clear out both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` environment variables.')
    print('Successfully logged out.')