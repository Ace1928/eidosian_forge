from __future__ import annotations
import inspect
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
@root_validator()
def validate_search_type(cls, values: Dict) -> Dict:
    """Validate search type."""
    if 'search_type' in values:
        search_type = values['search_type']
        if search_type not in ('similarity', 'mmr'):
            raise ValueError(f'search_type of {search_type} not allowed.')
    return values