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
def log_use_download_artifact(artifact: 'Artifact', alias: str, name: str, download_dir: str, failed_test_strings: List[str], add_extra_file: bool) -> Tuple[bool, Optional['Artifact'], List[str]]:
    with wandb.init(id=nice_id('log_artifact'), reinit=True, project=PROJECT_NAME, config={'test': 'artifact log'}) as log_art_run:
        if add_extra_file:
            with open('verify_2.txt', 'w') as f:
                f.write('2')
                f.close()
                artifact.add_file(f.name)
        try:
            log_art_run.log_artifact(artifact, aliases=alias)
        except Exception as e:
            failed_test_strings.append(f'Unable to log artifact. {e}')
            return (False, None, failed_test_strings)
    with wandb.init(id=nice_id('use_artifact'), project=PROJECT_NAME, config={'test': 'artifact use'}) as use_art_run:
        try:
            used_art = use_art_run.use_artifact(f'{name}:{alias}')
        except Exception as e:
            failed_test_strings.append(f'Unable to use artifact. {e}')
            return (False, None, failed_test_strings)
        try:
            used_art.download(root=download_dir)
        except Exception:
            failed_test_strings.append('Unable to download artifact. Check bucket permissions.')
            return (False, None, failed_test_strings)
    return (True, used_art, failed_test_strings)