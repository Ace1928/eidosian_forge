from __future__ import annotations
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
def upvote(self, query: str, document_id: int) -> None:
    """The retriever upweights the score of a document for a specific query.
        This is useful for fine-tuning the retriever to user behavior.

        Args:
            query: text to associate with `document_id`
            document_id: id of the document to associate query with.
        """
    self.db.text_to_result(query, document_id)