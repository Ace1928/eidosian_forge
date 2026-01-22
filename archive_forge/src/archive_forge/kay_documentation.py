from __future__ import annotations
from typing import Any, List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

        Create a KayRetriever given a Kay dataset id and a list of datasources.

        Args:
            dataset_id: A dataset id category in Kay, like "company"
            data_types: A list of datasources present within a dataset. For
                "company" the corresponding datasources could be
                ["10-K", "10-Q", "8-K", "PressRelease"].
            num_contexts: The number of documents to retrieve on each query.
                Defaults to 6.
        