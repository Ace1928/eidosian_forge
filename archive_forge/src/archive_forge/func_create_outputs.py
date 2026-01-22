from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from langchain_core.callbacks import (
from langchain_core.language_models import (
from langchain_core.load.dump import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseLLMOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.runnables import (
from langchain_core.runnables.configurable import DynamicRunnable
from langchain_core.utils.input import get_colored_text
from langchain.chains.base import Chain
def create_outputs(self, llm_result: LLMResult) -> List[Dict[str, Any]]:
    """Create outputs from response."""
    result = [{self.output_key: self.output_parser.parse_result(generation), 'full_generation': generation} for generation in llm_result.generations]
    if self.return_final_only:
        result = [{self.output_key: r[self.output_key]} for r in result]
    return result