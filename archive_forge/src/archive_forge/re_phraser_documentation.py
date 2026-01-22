import logging
from typing import List
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.chains.llm import LLMChain
Get relevated documents given a user question.

        Args:
            query: user question

        Returns:
            Relevant documents for re-phrased question
        