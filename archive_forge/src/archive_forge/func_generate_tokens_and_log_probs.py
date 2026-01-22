from __future__ import annotations
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from langchain_community.llms.openai import OpenAI
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain.chains.base import Chain
from langchain.chains.flare.prompts import (
from langchain.chains.llm import LLMChain
def generate_tokens_and_log_probs(self, _input: Dict[str, Any], *, run_manager: Optional[CallbackManagerForChainRun]=None) -> Tuple[Sequence[str], Sequence[float]]:
    llm_result = self.generate([_input], run_manager=run_manager)
    return self._extract_tokens_and_log_probs(llm_result.generations[0])