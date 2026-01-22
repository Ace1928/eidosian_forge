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
def upsert(self, entry: Dict[str, Any]) -> None:
    try:
        entry_id: str = entry['name']
    except KeyError:
        entry_id = entry['id']
    self._entries[entry_id] = wandb.util.merge_dicts(entry, self._entries[entry_id])