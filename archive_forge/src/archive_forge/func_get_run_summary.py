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
def get_run_summary(self, run_id: str, include_private: bool=False) -> Dict[str, Any]:
    mask_run = self.summary['__run_id'] == run_id
    run_summary = self.summary[mask_run]
    ret = (run_summary.filter(regex='^[^_]', axis=1) if not include_private else run_summary).to_dict(orient='records')
    return ret[0] if len(ret) > 0 else {}