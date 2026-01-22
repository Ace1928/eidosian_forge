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
def push_success(self, artifact_id: str, save_name: str) -> None:
    """Notification that a file upload has been successfully completed.

        Arguments:
            artifact_id: ID of artifact
            save_name: saved name of the uploaded file
        """
    self._queue.put(self.PushSuccess(artifact_id, save_name))