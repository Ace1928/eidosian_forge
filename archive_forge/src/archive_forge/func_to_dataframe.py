from __future__ import annotations
import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import (
from langchain_core._api import warn_deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables import config as runnable_config
from langchain_core.runnables import utils as runnable_utils
from langchain_core.tracers.evaluation import (
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client
from langsmith.env import get_git_info, get_langchain_env_var_metadata
from langsmith.evaluation import (
from langsmith.evaluation import (
from langsmith.run_helpers import as_runnable, is_traceable_function
from langsmith.schemas import Dataset, DataType, Example, Run, TracerSession
from langsmith.utils import LangSmithError
from requests import HTTPError
from typing_extensions import TypedDict
from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain
from langchain.evaluation.loading import load_evaluator
from langchain.evaluation.schema import (
from langchain.smith import evaluation as smith_eval
from langchain.smith.evaluation import config as smith_eval_config
from langchain.smith.evaluation import name_generation, progress
def to_dataframe(self) -> pd.DataFrame:
    """Convert the results to a dataframe."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError('Pandas is required to convert the results to a dataframe. to install pandas, run `pip install pandas`.') from e
    indices = []
    records = []
    for example_id, result in self['results'].items():
        feedback = result['feedback']
        output_ = result.get('output')
        if isinstance(output_, dict):
            output = {f'outputs.{k}': v for k, v in output_.items()}
        elif output_ is None:
            output = {}
        else:
            output = {'output': output_}
        r = {**{f'inputs.{k}': v for k, v in result['input'].items()}, **output}
        if 'reference' in result:
            if isinstance(result['reference'], dict):
                r.update({f'reference.{k}': v for k, v in result['reference'].items()})
            else:
                r['reference'] = result['reference']
        r.update({**{f'feedback.{f.key}': f.score for f in feedback}, 'error': result.get('Error'), 'execution_time': result['execution_time'], 'run_id': result.get('run_id')})
        records.append(r)
        indices.append(example_id)
    return pd.DataFrame(records, index=indices)