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
def resolve_preempting(request_data: Dict[str, Any], response_data: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(request_data, dict):
        return None
    query = 'preempting' in request_data
    if query:
        name = kwargs.get('path').split('/')[2]
        post_processed_data = {'name': name, 'preempting': [request_data['preempting']]}
        return post_processed_data
    return None