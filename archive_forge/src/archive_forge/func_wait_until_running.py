import json
import os
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.errors import CommError
from wandb.sdk.artifacts.artifact_state import ArtifactState
from wandb.sdk.data_types._dtypes import InvalidType, Type, TypeRegistry
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import (
@normalize_exceptions
def wait_until_running(self):
    if self._run is not None:
        return self._run
    while True:
        time.sleep(2)
        item = self._get_item()
        if item and item['associatedRunId'] is not None:
            try:
                self._run = public.Run(self.client, self._entity, self.project, item['associatedRunId'], None)
                self._run_id = item['associatedRunId']
                return self._run
            except ValueError as e:
                print(e)
        elif item:
            wandb.termlog('Waiting for run to start')
        time.sleep(3)