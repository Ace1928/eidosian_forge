import getpass
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import click
import requests
from wandb_gql import gql
import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.lib import runid
from ...apis.internal import Api
def check_wandb_version(api: Api) -> None:
    print('Checking wandb package version is up to date'.ljust(72, '.'), end='')
    _, server_info = api.viewer_server_info()
    fail_string = None
    warning = False
    max_cli_version = server_info.get('cliVersionInfo', {}).get('max_cli_version', None)
    min_cli_version = server_info.get('cliVersionInfo', {}).get('min_cli_version', '0.0.1')
    from wandb.util import parse_version
    if parse_version(wandb.__version__) < parse_version(min_cli_version):
        fail_string = 'wandb version out of date, please run pip install --upgrade wandb=={}'.format(max_cli_version)
    elif parse_version(wandb.__version__) > parse_version(max_cli_version):
        fail_string = f"wandb version is not supported by your local installation. This could cause some issues. If you're having problems try: please run `pip install --upgrade wandb=={max_cli_version}`"
        warning = True
    print_results(fail_string, warning)