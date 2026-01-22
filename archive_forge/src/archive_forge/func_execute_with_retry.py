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
def execute_with_retry(self, _run_manager: CallbackManagerForChainRun, callbacks: CallbackManager, generated_gremlin: str) -> List[Any]:
    try:
        return self.execute_query(generated_gremlin)
    except Exception as e:
        retries = 0
        error_message = str(e)
        self.log_invalid_query(_run_manager, generated_gremlin, error_message)
        while retries < self.max_fix_retries:
            try:
                fix_chain_result = self.gremlin_fix_chain.invoke({'error_message': error_message, 'generated_sparql': generated_gremlin, 'schema': self.schema}, callbacks=callbacks)
                fixed_gremlin = fix_chain_result[self.gremlin_fix_chain.output_key]
                return self.execute_query(fixed_gremlin)
            except Exception as e:
                retries += 1
                parse_exception = str(e)
                self.log_invalid_query(_run_manager, fixed_gremlin, parse_exception)
    raise ValueError('The generated Gremlin query is invalid.')