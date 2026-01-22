import logging
import re
from typing import List, Optional
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.llms import LlamaCpp
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
def clean_search_query(self, query: str) -> str:
    if query[0].isdigit():
        first_quote_pos = query.find('"')
        if first_quote_pos != -1:
            query = query[first_quote_pos + 1:]
            if query.endswith('"'):
                query = query[:-1]
    return query.strip()