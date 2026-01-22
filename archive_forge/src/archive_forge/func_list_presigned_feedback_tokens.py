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
def list_presigned_feedback_tokens(self, run_id: ID_TYPE, *, limit: Optional[int]=None) -> Iterator[ls_schemas.FeedbackIngestToken]:
    """List the feedback ingest tokens for a run.

        Args:
            run_id: The ID of the run to filter by.
            limit: The maximum number of tokens to return.

        Yields:
            FeedbackIngestToken
                The feedback ingest tokens.
        """
    params = {'run_id': _as_uuid(run_id, 'run_id'), 'limit': min(limit, 100) if limit is not None else 100}
    for i, token in enumerate(self._get_paginated_list('/feedback/tokens', params=params)):
        yield ls_schemas.FeedbackIngestToken(**token)
        if limit is not None and i + 1 >= limit:
            break