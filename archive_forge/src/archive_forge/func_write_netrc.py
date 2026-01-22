import os
import stat
import sys
import textwrap
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union
from urllib.parse import urlparse
import click
import requests.utils
import wandb
from wandb.apis import InternalApi
from wandb.errors import term
from wandb.util import _is_databricks, isatty, prompt_choices
from .wburls import wburls
def write_netrc(host: str, entity: str, key: str) -> Optional[bool]:
    """Add our host and key to .netrc."""
    _, key_suffix = key.split('-', 1) if '-' in key else ('', key)
    if len(key_suffix) != 40:
        wandb.termerror('API-key must be exactly 40 characters long: {} ({} chars)'.format(key_suffix, len(key_suffix)))
        return None
    try:
        normalized_host = urlparse(host).netloc.split(':')[0]
        netrc_path = get_netrc_file_path()
        wandb.termlog(f'Appending key for {normalized_host} to your netrc file: {netrc_path}')
        machine_line = f'machine {normalized_host}'
        orig_lines = None
        try:
            with open(netrc_path) as f:
                orig_lines = f.read().strip().split('\n')
        except OSError:
            pass
        with open(netrc_path, 'w') as f:
            if orig_lines:
                skip = 0
                for line in orig_lines:
                    if line == 'machine ' or machine_line in line:
                        skip = 2
                    elif skip:
                        skip -= 1
                    else:
                        f.write('%s\n' % line)
            f.write(textwrap.dedent('            machine {host}\n              login {entity}\n              password {key}\n            ').format(host=normalized_host, entity=entity, key=key))
        os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR)
        return True
    except OSError:
        wandb.termerror(f'Unable to read {netrc_path}')
        return None