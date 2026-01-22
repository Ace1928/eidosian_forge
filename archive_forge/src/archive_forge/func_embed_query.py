from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra
from langchain.chains.base import Chain
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.chains.llm import LLMChain
def embed_query(self, text: str) -> List[float]:
    """Generate a hypothetical document and embedded it."""
    var_name = self.llm_chain.input_keys[0]
    result = self.llm_chain.generate([{var_name: text}])
    documents = [generation.text for generation in result.generations[0]]
    embeddings = self.embed_documents(documents)
    return self.combine_embeddings(embeddings)