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
def upvote_batch(self, query_id_pairs: List[Tuple[str, int]]) -> None:
    """Given a batch of (query, document id) pairs, the retriever upweights
        the scores of the document for the corresponding queries.
        This is useful for fine-tuning the retriever to user behavior.

        Args:
            query_id_pairs: list of (query, document id) pairs. For each pair in
            this list, the model will upweight the document id for the query.
        """
    self.db.text_to_result_batch(query_id_pairs)