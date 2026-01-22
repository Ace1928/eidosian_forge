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
def create_presigned_feedback_token(self, run_id: ID_TYPE, feedback_key: str, *, expiration: Optional[datetime.datetime | datetime.timedelta]=None, feedback_config: Optional[ls_schemas.FeedbackConfig]=None) -> ls_schemas.FeedbackIngestToken:
    """Create a pre-signed URL to send feedback data to.

        This is useful for giving browser-based clients a way to upload
        feedback data directly to LangSmith without accessing the
        API key.

        Args:
            run_id:
            feedback_key:
            expiration: The expiration time of the pre-signed URL.
                Either a datetime or a timedelta offset from now.
                Default to 3 hours.
            feedback_config: FeedbackConfig or None.
                If creating a feedback_key for the first time,
                this defines how the metric should be interpreted,
                such as a continuous score (w/ optional bounds),
                or distribution over categorical values.

        Returns:
            The pre-signed URL for uploading feedback data.
        """
    body: Dict[str, Any] = {'run_id': run_id, 'feedback_key': feedback_key, 'feedback_config': feedback_config}
    if expiration is None:
        body['expires_in'] = ls_schemas.TimeDeltaInput(days=0, hours=3, minutes=0)
    elif isinstance(expiration, datetime.datetime):
        body['expires_at'] = expiration.isoformat()
    elif isinstance(expiration, datetime.timedelta):
        body['expires_in'] = ls_schemas.TimeDeltaInput(days=expiration.days, hours=expiration.seconds // 3600, minutes=expiration.seconds // 60 % 60)
    else:
        raise ValueError(f'Unknown expiration type: {type(expiration)}')
    response = self.request_with_retries('POST', '/feedback/tokens', data=_dumps_json(body))
    ls_utils.raise_for_status_with_text(response)
    return ls_schemas.FeedbackIngestToken(**response.json())