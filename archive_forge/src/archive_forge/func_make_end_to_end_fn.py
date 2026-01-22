from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal
import httpx
import huggingface_hub
from huggingface_hub import CommitOperationAdd, SpaceHardware, SpaceStage
from huggingface_hub.utils import (
from packaging import version
from gradio_client import utils
from gradio_client.compatibility import EndpointV3Compatibility
from gradio_client.data_classes import ParameterInfo
from gradio_client.documentation import document
from gradio_client.exceptions import AuthenticationError
from gradio_client.utils import (
def make_end_to_end_fn(self, helper: Communicator | None=None):
    _predict = self.make_predict(helper)

    def _inner(*data):
        if not self.is_valid:
            raise utils.InvalidAPIEndpointError()
        data = self.insert_empty_state(*data)
        data = self.process_input_files(*data)
        predictions = _predict(*data)
        predictions = self.process_predictions(*predictions)
        if helper:
            with helper.lock:
                if not helper.job.outputs:
                    helper.job.outputs.append(predictions)
        return predictions
    return _inner