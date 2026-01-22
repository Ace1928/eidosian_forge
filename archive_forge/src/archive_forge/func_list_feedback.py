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
def list_feedback(self, *, run_ids: Optional[Sequence[ID_TYPE]]=None, feedback_key: Optional[Sequence[str]]=None, feedback_source_type: Optional[Sequence[ls_schemas.FeedbackSourceType]]=None, limit: Optional[int]=None, **kwargs: Any) -> Iterator[ls_schemas.Feedback]:
    """List the feedback objects on the LangSmith API.

        Parameters
        ----------
        run_ids : List[str or UUID] or None, default=None
            The IDs of the runs to filter by.
        feedback_key: List[str] or None, default=None
            The feedback key(s) to filter by. Example: 'correctness'
            The query performs a union of all feedback keys.
        feedback_source_type: List[FeedbackSourceType] or None, default=None
            The type of feedback source, such as model
            (for model-generated feedback) or API.
        limit : int or None, default=None
        **kwargs : Any
            Additional keyword arguments.

        Yields:
        ------
        Feedback
            The feedback objects.
        """
    params: dict = {'run': run_ids, 'limit': min(limit, 100) if limit is not None else 100, **kwargs}
    if feedback_key is not None:
        params['key'] = feedback_key
    if feedback_source_type is not None:
        params['source'] = feedback_source_type
    for i, feedback in enumerate(self._get_paginated_list('/feedback', params=params)):
        yield ls_schemas.Feedback(**feedback)
        if limit is not None and i + 1 >= limit:
            break