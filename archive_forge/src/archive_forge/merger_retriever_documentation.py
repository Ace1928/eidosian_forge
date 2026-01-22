import asyncio
from typing import List
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

        Asynchronously merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        