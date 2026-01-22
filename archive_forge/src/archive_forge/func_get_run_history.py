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
def get_run_history(self, run_id: str, include_private: bool=False) -> pd.DataFrame:
    mask_run = self.history['__run_id'] == run_id
    run_history = self.history[mask_run]
    return run_history.filter(regex='^[^_]', axis=1) if not include_private else run_history