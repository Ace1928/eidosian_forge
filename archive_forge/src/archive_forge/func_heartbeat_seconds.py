import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
@property
def heartbeat_seconds(self) -> Union[int, float]:
    heartbeat_seconds: Union[int, float] = self._api.dynamic_settings['heartbeat_seconds']
    return heartbeat_seconds