from __future__ import annotations
import collections
import concurrent.futures as cf
import datetime
import functools
import itertools
import logging
import pathlib
import threading
import uuid
from contextvars import copy_context
from typing import (
from requests import HTTPError
from typing_extensions import TypedDict
import langsmith
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import run_trees, schemas
from langsmith import utils as ls_utils
from langsmith.evaluation.evaluator import (
from langsmith.evaluation.integrations import LangChainStringEvaluator
def with_predictions(self, target: TARGET_T, /, max_concurrency: Optional[int]=None) -> _ExperimentManager:
    """Lazily apply the target function to the experiment."""
    context = copy_context()
    _experiment_results = context.run(self._predict, target, max_concurrency=max_concurrency)
    r1, r2 = itertools.tee(_experiment_results, 2)
    return _ExperimentManager((pred['example'] for pred in r1), experiment=self._experiment, metadata=self._metadata, client=self.client, runs=(pred['run'] for pred in r2))