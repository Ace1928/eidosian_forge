from __future__ import annotations
import inspect
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.chains import ReduceDocumentsChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
@root_validator(pre=True)
def validate_naming(cls, values: Dict) -> Dict:
    """Fix backwards compatibility in naming."""
    if 'combine_document_chain' in values:
        values['combine_documents_chain'] = values.pop('combine_document_chain')
    return values