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
def update_example(self, example_id: ID_TYPE, *, inputs: Optional[Dict[str, Any]]=None, outputs: Optional[Mapping[str, Any]]=None, metadata: Optional[Dict]=None, dataset_id: Optional[ID_TYPE]=None) -> Dict[str, Any]:
    """Update a specific example.

        Parameters
        ----------
        example_id : str or UUID
            The ID of the example to update.
        inputs : Dict[str, Any] or None, default=None
            The input values to update.
        outputs : Mapping[str, Any] or None, default=None
            The output values to update.
        metadata : Dict or None, default=None
            The metadata to update.
        dataset_id : UUID or None, default=None
            The ID of the dataset to update.

        Returns:
        -------
        Dict[str, Any]
            The updated example.
        """
    example = ls_schemas.ExampleUpdate(inputs=inputs, outputs=outputs, dataset_id=dataset_id, metadata=metadata)
    response = self.session.patch(f'{self.api_url}/examples/{_as_uuid(example_id, 'example_id')}', headers={**self._headers, 'Content-Type': 'application/json'}, data=example.json(exclude_none=True))
    ls_utils.raise_for_status_with_text(response)
    return response.json()