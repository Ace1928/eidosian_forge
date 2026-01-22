from __future__ import annotations
import functools
import uuid
import warnings
from itertools import islice
from operator import itemgetter
from typing import (
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def sync_call_fallback(method: Callable) -> Callable:
    """
    Decorator to call the synchronous method of the class if the async method is not
    implemented. This decorator might be only used for the methods that are defined
    as async in the class.
    """

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await method(self, *args, **kwargs)
        except NotImplementedError:
            return await run_in_executor(None, getattr(self, method.__name__[1:]), *args, **kwargs)
    return wrapper