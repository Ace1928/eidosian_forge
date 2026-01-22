from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
Load the retriever from local storage.

        Args:
            folder_path: Folder path to load from.
            allow_dangerous_deserialization: Whether to allow dangerous deserialization.
                Defaults to False.
                The deserialization relies on .joblib and .pkl files, which can be
                modified to deliver a malicious payload that results in execution of
                arbitrary code on your machine. You will need to set this to `True` to
                use deserialization. If you do this, make sure you trust the source of
                the file.
            file_name: File name to load from. Defaults to "tfidf_vectorizer".

        Returns:
            TFIDFRetriever: Loaded retriever.
        