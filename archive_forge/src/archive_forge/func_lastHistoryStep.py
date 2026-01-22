import json
import os
import tempfile
import time
import urllib
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import env, util
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.internal import Api as InternalApi
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.apis.public.const import RETRY_TIMEDELTA
from wandb.sdk.lib import ipython, json_util, runid
from wandb.sdk.lib.paths import LogicalPath
@property
def lastHistoryStep(self):
    query = gql('\n        query RunHistoryKeys($project: String!, $entity: String!, $name: String!) {\n            project(name: $project, entityName: $entity) {\n                run(name: $name) { historyKeys }\n            }\n        }\n        ')
    response = self._exec(query)
    if response is None or response.get('project') is None or response['project'].get('run') is None or (response['project']['run'].get('historyKeys') is None):
        return -1
    history_keys = response['project']['run']['historyKeys']
    return history_keys['lastStep'] if 'lastStep' in history_keys else -1