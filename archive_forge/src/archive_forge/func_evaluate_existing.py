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
def evaluate_existing(experiment: Union[str, uuid.UUID], /, evaluators: Optional[Sequence[EVALUATOR_T]]=None, summary_evaluators: Optional[Sequence[SUMMARY_EVALUATOR_T]]=None, metadata: Optional[dict]=None, max_concurrency: Optional[int]=None, client: Optional[langsmith.Client]=None, load_nested: bool=False, blocking: bool=True) -> ExperimentResults:
    """Evaluate existing experiment runs.

    Args:
        experiment (Union[str, uuid.UUID]): The identifier of the experiment to evaluate.
        data (DATA_T): The data to use for evaluation.
        evaluators (Optional[Sequence[EVALUATOR_T]]): Optional sequence of evaluators to use for individual run evaluation.
        summary_evaluators (Optional[Sequence[SUMMARY_EVALUATOR_T]]): Optional sequence of evaluators
            to apply over the entire dataset.
        metadata (Optional[dict]): Optional metadata to include in the evaluation results.
        max_concurrency (Optional[int]): Optional maximum number of concurrent evaluations.
        client (Optional[langsmith.Client]): Optional Langsmith client to use for evaluation.
        load_nested: Whether to load all child runs for the experiment.
            Default is to only load the top-level root runs.
        blocking (bool): Whether to block until evaluation is complete.

    Returns:
        ExperimentResults: The evaluation results.

    Environment:
        - LANGSMITH_TEST_CACHE: If set, API calls will be cached to disk to save time and
            cost during testing. Recommended to commit the cache files to your repository
            for faster CI/CD runs.
            Requires the 'langsmith[vcr]' package to be installed.

    Examples:
        >>> from langsmith.evaluation import evaluate, evaluate_existing
        >>> dataset_name = "Evaluate Examples"
        >>> def predict(inputs: dict) -> dict:
        ...     # This can be any function or just an API call to your app.
        ...     return {"output": "Yes"}
        >>> # First run inference on the dataset
        ... results = evaluate(
        ...     predict,
        ...     data=dataset_name,
        ... )  # doctest: +ELLIPSIS
        View the evaluation results for experiment:...
        >>> # Then apply evaluators to the experiment
        ... def accuracy(run: Run, example: Example):
        ...     # Row-level evaluator for accuracy.
        ...     pred = run.outputs["output"]
        ...     expected = example.outputs["answer"]
        ...     return {"score": expected.lower() == pred.lower()}
        >>> def precision(runs: Sequence[Run], examples: Sequence[Example]):
        ...     # Experiment-level evaluator for precision.
        ...     # TP / (TP + FP)
        ...     predictions = [run.outputs["output"].lower() for run in runs]
        ...     expected = [example.outputs["answer"].lower() for example in examples]
        ...     # yes and no are the only possible answers
        ...     tp = sum([p == e for p, e in zip(predictions, expected) if p == "yes"])
        ...     fp = sum([p == "yes" and e == "no" for p, e in zip(predictions, expected)])
        ...     return {"score": tp / (tp + fp)}
        >>> experiment_name = (
        ...     results.experiment_name
        ... )  # Can use the returned experiment name
        >>> experiment_name = "My Experiment:64e6e91"  # Or manually specify
        >>> results = evaluate_existing(
        ...     experiment_name,
        ...     summary_evaluators=[precision],
        ... )  # doctest: +ELLIPSIS
        View the evaluation results for experiment:...
    """
    client = client or langsmith.Client()
    project = _load_experiment(experiment, client)
    runs = _load_traces(experiment, client, load_nested=load_nested)
    data = list(client.list_examples(dataset_id=project.reference_dataset_id, as_of=project.metadata.get('dataset_version')))
    runs = sorted(runs, key=lambda r: str(r.reference_example_id))
    data = sorted(data, key=lambda d: str(d.id))
    return _evaluate(runs, data=data, evaluators=evaluators, summary_evaluators=summary_evaluators, metadata=metadata, max_concurrency=max_concurrency, client=client, blocking=blocking)