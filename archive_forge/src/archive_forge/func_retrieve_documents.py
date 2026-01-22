import asyncio
import logging
from typing import List, Optional, Sequence
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.chains.llm import LLMChain
def retrieve_documents(self, queries: List[str], run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
    """Run all LLM generated queries.

        Args:
            queries: query list

        Returns:
            List of retrieved Documents
        """
    documents = []
    for query in queries:
        docs = self.retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
        documents.extend(docs)
    return documents