import re
import subprocess
from typing import List, Optional
from ..constants import ENDPOINT
from ._subprocess import run_interactive_subprocess, run_subprocess
def unset_git_credential(username: str='hf_user', folder: Optional[str]=None) -> None:
    """Erase credentials from git credential for HF Hub registry.

    Credentials are erased from the configured helpers (store, cache, macOS
    keychain,...), if any. If `username` is not provided, any credential configured for
    HF Hub endpoint is erased.
    Calls "`git credential erase`" internally. See https://git-scm.com/docs/git-credential.

    Args:
        username (`str`, defaults to `"hf_user"`):
            A git username. Defaults to `"hf_user"`, the default user used in the Hub.
        folder (`str`, *optional*):
            The folder in which to check the configured helpers.
    """
    with run_interactive_subprocess('git credential reject', folder=folder) as (stdin, _):
        standard_input = f'url={ENDPOINT}\n'
        if username is not None:
            standard_input += f'username={username.lower()}\n'
        standard_input += '\n'
        stdin.write(standard_input)
        stdin.flush()