from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.load.dump import dumpd
from langchain_core.load.load import load
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage, get_buffer_string, messages_from_dict
from langsmith import EvaluationResult, RunEvaluator
from langsmith.schemas import DataType, Example, Run
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator
from langchain.schema import RUN_KEY
@classmethod
def from_run_and_data_type(cls, evaluator: StringEvaluator, run_type: str, data_type: DataType, input_key: Optional[str]=None, prediction_key: Optional[str]=None, reference_key: Optional[str]=None, tags: Optional[List[str]]=None) -> StringRunEvaluatorChain:
    """
        Create a StringRunEvaluatorChain from an evaluator and the run and dataset types.

        This method provides an easy way to instantiate a StringRunEvaluatorChain, by
        taking an evaluator and information about the type of run and the data.
        The method supports LLM and chain runs.

        Args:
            evaluator (StringEvaluator): The string evaluator to use.
            run_type (str): The type of run being evaluated.
                Supported types are LLM and Chain.
            data_type (DataType): The type of dataset used in the run.
            input_key (str, optional): The key used to map the input from the run.
            prediction_key (str, optional): The key used to map the prediction from the run.
            reference_key (str, optional): The key used to map the reference from the dataset.
            tags (List[str], optional): List of tags to attach to the evaluation chain.

        Returns:
            StringRunEvaluatorChain: The instantiated evaluation chain.

        Raises:
            ValueError: If the run type is not supported, or if the evaluator requires a
                reference from the dataset but the reference key is not provided.

        """
    if run_type == 'llm':
        run_mapper: StringRunMapper = LLMStringRunMapper()
    elif run_type == 'chain':
        run_mapper = ChainStringRunMapper(input_key=input_key, prediction_key=prediction_key)
    else:
        raise ValueError(f"Unsupported run type {run_type}. Expected one of 'llm' or 'chain'.")
    if reference_key is not None or data_type in (DataType.llm, DataType.chat) or evaluator.requires_reference:
        example_mapper = StringExampleMapper(reference_key=reference_key)
    elif evaluator.requires_reference:
        raise ValueError(f'Evaluator {evaluator.evaluation_name} requires a reference example from the dataset. Please specify the reference key from amongst the dataset outputs keys.')
    else:
        example_mapper = None
    return cls(name=evaluator.evaluation_name, run_mapper=run_mapper, example_mapper=example_mapper, string_evaluator=evaluator, tags=tags)