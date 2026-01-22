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
def verify_manifest(downloaded_manifest: Dict[str, Any], computed_manifest: Dict[str, Any], fails_list: List[str]) -> None:
    try:
        for key in computed_manifest.keys():
            assert computed_manifest[key]['digest'] == downloaded_manifest[key]['digest']
            assert computed_manifest[key]['size'] == downloaded_manifest[key]['size']
    except AssertionError:
        fails_list.append('Artifact manifest does not appear as expected. Contact W&B for support.')