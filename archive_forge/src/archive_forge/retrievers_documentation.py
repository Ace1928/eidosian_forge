from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.runnables import (
from langchain_core.runnables.config import run_in_executor
Asynchronously get documents relevant to a query.

        Users should favor using `.ainvoke` or `.abatch` rather than
        `aget_relevant_documents directly`.

        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
            tags: Optional list of tags associated with the retriever. Defaults to None
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            metadata: Optional metadata associated with the retriever. Defaults to None
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            run_name: Optional name for the run.

        Returns:
            List of relevant documents
        