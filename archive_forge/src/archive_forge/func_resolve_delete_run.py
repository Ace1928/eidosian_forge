import dataclasses
import json
import logging
import socket
import sys
import threading
import traceback
import urllib.parse
from collections import defaultdict, deque
from copy import deepcopy
from typing import (
import flask
import pandas as pd
import requests
import responses
import wandb
import wandb.util
from wandb.sdk.lib.timer import Timer
@staticmethod
def resolve_delete_run(request_data: Dict[str, Any], response_data: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(request_data, dict) or not isinstance(response_data, dict):
        return None
    query = 'query' in request_data and 'deleteRun' in request_data['query']
    if query:
        data = {k: v for k, v in request_data['variables'].items() if v is not None}
        data.update(response_data['data']['deleteRun'])
        return data
    return None