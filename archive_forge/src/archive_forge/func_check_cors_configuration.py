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
def check_cors_configuration(url: str, origin: str) -> None:
    print('Checking CORs configuration of the bucket'.ljust(72, '.'), end='')
    fail_string = None
    res_get = requests.options(url, headers={'Origin': origin, 'Access-Control-Request-Method': 'GET'})
    if res_get.headers.get('Access-Control-Allow-Origin') is None:
        fail_string = f'Your object store does not have a valid CORs configuration, you must allow GET and PUT to Origin: {origin}'
    print_results(fail_string, True)