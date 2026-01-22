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
def create_examples(self, *, inputs: Sequence[Mapping[str, Any]], outputs: Optional[Sequence[Optional[Mapping[str, Any]]]]=None, metadata: Optional[Sequence[Optional[Mapping[str, Any]]]]=None, source_run_ids: Optional[Sequence[Optional[ID_TYPE]]]=None, ids: Optional[Sequence[Optional[ID_TYPE]]]=None, dataset_id: Optional[ID_TYPE]=None, dataset_name: Optional[str]=None, **kwargs: Any) -> None:
    """Create examples in a dataset.

        Parameters
        ----------
        inputs : Sequence[Mapping[str, Any]]
            The input values for the examples.
        outputs : Optional[Sequence[Optional[Mapping[str, Any]]]], default=None
            The output values for the examples.
        metadata : Optional[Sequence[Optional[Mapping[str, Any]]]], default=None
            The metadata for the examples.
        source_run_ids : Optional[Sequence[Optional[ID_TYPE]]], default=None
                The IDs of the source runs associated with the examples.
        ids : Optional[Sequence[ID_TYPE]], default=None
            The IDs of the examples.
        dataset_id : Optional[ID_TYPE], default=None
            The ID of the dataset to create the examples in.
        dataset_name : Optional[str], default=None
            The name of the dataset to create the examples in.

        Returns:
        -------
        None

        Raises:
        ------
        ValueError
            If both `dataset_id` and `dataset_name` are `None`.
        """
    if dataset_id is None and dataset_name is None:
        raise ValueError('Either dataset_id or dataset_name must be provided.')
    if dataset_id is None:
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
    examples = [{'inputs': in_, 'outputs': out_, 'dataset_id': dataset_id, 'metadata': metadata_, 'id': id_, 'source_run_id': source_run_id_} for in_, out_, metadata_, id_, source_run_id_ in zip(inputs, outputs or [None] * len(inputs), metadata or [None] * len(inputs), ids or [None] * len(inputs), source_run_ids or [None] * len(inputs))]
    response = self.session.post(f'{self.api_url}/examples/bulk', headers={**self._headers, 'Content-Type': 'application/json'}, data=_dumps_json(examples))
    ls_utils.raise_for_status_with_text(response)