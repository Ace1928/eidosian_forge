import asyncio
from typing import List
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
def merge_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
    """
        Merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """
    retriever_docs = [retriever.get_relevant_documents(query, callbacks=run_manager.get_child('retriever_{}'.format(i + 1))) for i, retriever in enumerate(self.retrievers)]
    merged_documents = []
    max_docs = max(map(len, retriever_docs), default=0)
    for i in range(max_docs):
        for retriever, doc in zip(self.retrievers, retriever_docs):
            if i < len(doc):
                merged_documents.append(doc[i])
    return merged_documents