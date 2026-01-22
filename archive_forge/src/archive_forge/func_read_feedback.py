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
def read_feedback(self, feedback_id: ID_TYPE) -> ls_schemas.Feedback:
    """Read a feedback from the LangSmith API.

        Parameters
        ----------
        feedback_id : str or UUID
            The ID of the feedback to read.

        Returns:
        -------
        Feedback
            The feedback.
        """
    response = self.request_with_retries('GET', f'/feedback/{_as_uuid(feedback_id, 'feedback_id')}')
    return ls_schemas.Feedback(**response.json())