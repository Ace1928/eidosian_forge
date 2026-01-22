from typing import Any, List, Optional, Type, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseLLMOutputParser
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import get_llm_kwargs
def create_qa_with_sources_chain(llm: BaseLanguageModel, verbose: bool=False, **kwargs: Any) -> LLMChain:
    """Create a question answering chain that returns an answer with sources.

    Args:
        llm: Language model to use for the chain.
        verbose: Whether to print the details of the chain
        **kwargs: Keyword arguments to pass to `create_qa_with_structure_chain`.

    Returns:
        Chain (LLMChain) that can be used to answer questions with citations.
    """
    return create_qa_with_structure_chain(llm, AnswerWithSources, verbose=verbose, **kwargs)