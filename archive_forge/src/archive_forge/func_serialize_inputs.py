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
def serialize_inputs(self, inputs: Dict) -> str:
    if 'prompts' in inputs:
        input_ = '\n\n'.join(inputs['prompts'])
    elif 'prompt' in inputs:
        input_ = inputs['prompt']
    elif 'messages' in inputs:
        input_ = self.serialize_chat_messages(inputs['messages'])
    else:
        raise ValueError('LLM Run must have either messages or prompts as inputs.')
    return input_