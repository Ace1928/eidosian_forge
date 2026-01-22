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
@classmethod
def from_scratch(cls, thirdai_key: Optional[str]=None, **model_kwargs: dict) -> NeuralDBRetriever:
    """
        Create a NeuralDBRetriever from scratch.

        To use, set the ``THIRDAI_KEY`` environment variable with your ThirdAI
        API key, or pass ``thirdai_key`` as a named parameter.

        Example:
            .. code-block:: python

                from langchain_community.retrievers import NeuralDBRetriever

                retriever = NeuralDBRetriever.from_scratch(
                    thirdai_key="your-thirdai-key",
                )

                retriever.insert([
                    "/path/to/doc.pdf",
                    "/path/to/doc.docx",
                    "/path/to/doc.csv",
                ])

                documents = retriever.get_relevant_documents("AI-driven music therapy")
        """
    NeuralDBRetriever._verify_thirdai_library(thirdai_key)
    from thirdai import neural_db as ndb
    return cls(thirdai_key=thirdai_key, db=ndb.NeuralDB(**model_kwargs))