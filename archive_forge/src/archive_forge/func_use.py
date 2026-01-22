from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def use(self, table_name: str, **kwargs: Any) -> bool:
    """Use the specified table. Don't know the tables, please invoke list_tables."""
    if self.awadb_client is None:
        return False
    ret = self.awadb_client.Use(table_name)
    if ret:
        self.using_table_name = table_name
    return ret