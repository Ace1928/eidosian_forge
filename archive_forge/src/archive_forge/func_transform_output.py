import io
import json
from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, TypeVar, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
@abstractmethod
def transform_output(self, output: bytes) -> OUTPUT_TYPE:
    """Transforms the output from the model to string that
        the LLM class expects.
        """