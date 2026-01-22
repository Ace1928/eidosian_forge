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
def get_run_stats(self, run_id: str) -> pd.DataFrame:
    mask_run = self.events['__run_id'] == run_id
    run_stats = self.events[mask_run]
    return run_stats