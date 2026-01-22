from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

        Creates an instance of the VST class from the given texts.

        Args:
            texts (List[str]): List of texts to be added.
            embedding (Embeddings): Embedding model for the texts.
            metadatas (List[dict], optional):
            List of metadata dictionaries for each text.Defaults to None.
            table_name (str): Name of the table. Defaults to "test".
            database_name (str): Name of the database. Defaults to "default".
            connection_args (dict[str, Any]): Connection parameters.
            Defaults to DEFAULT_HIPPO_CONNECTION.
            index_params (dict): Indexing parameters. Defaults to None.
            search_params (dict): Search parameters. Defaults to an empty dictionary.
            drop_old (bool): Whether to drop the old collection. Defaults to False.
            kwargs: Other arguments.

        Returns:
            Hippo: An instance of the VST class.
        