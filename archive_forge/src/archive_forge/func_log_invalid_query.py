from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_community.graphs import GremlinGraph
from langchain_core.callbacks.manager import CallbackManager, CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import (
from langchain.chains.llm import LLMChain
def log_invalid_query(self, _run_manager: CallbackManagerForChainRun, generated_query: str, error_message: str) -> None:
    _run_manager.on_text('Invalid Gremlin query: ', end='\n', verbose=self.verbose)
    _run_manager.on_text(generated_query, color='red', end='\n', verbose=self.verbose)
    _run_manager.on_text('Gremlin Query Parse Error: ', end='\n', verbose=self.verbose)
    _run_manager.on_text(error_message, color='red', end='\n\n', verbose=self.verbose)