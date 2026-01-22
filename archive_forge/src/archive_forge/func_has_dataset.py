from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def has_dataset(self, *, dataset_name: Optional[str]=None, dataset_id: Optional[str]=None) -> bool:
    """Check whether a dataset exists in your tenant.

        Parameters
        ----------
        dataset_name : str or None, default=None
            The name of the dataset to check.
        dataset_id : str or None, default=None
            The ID of the dataset to check.

        Returns:
        -------
        bool
            Whether the dataset exists.
        """
    try:
        self.read_dataset(dataset_name=dataset_name, dataset_id=dataset_id)
        return True
    except ls_utils.LangSmithNotFoundError:
        return False