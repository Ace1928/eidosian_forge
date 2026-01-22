import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
def attempt_kaggle_load_ipynb():
    kaggle = wandb.util.get_module('kaggle_session')
    if kaggle:
        try:
            client = kaggle.UserSessionClient()
            parsed = json.loads(client.get_exportable_ipynb()['source'])
            parsed['metadata']['name'] = 'kaggle.ipynb'
            return parsed
        except Exception:
            logger.exception('Unable to load kaggle notebook')
            return None