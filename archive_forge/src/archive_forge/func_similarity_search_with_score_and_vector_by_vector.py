from __future__ import annotations
import importlib.util
import json
import re
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
def similarity_search_with_score_and_vector_by_vector(self, embedding: List[float], k: int=4, filter: Optional[dict]=None) -> List[Tuple[Document, float, List[float]]]:
    """Return docs most similar to the given embedding.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query and
            score and the document's embedding vector for each
        """
    result = []
    k = HanaDB._sanitize_int(k)
    embedding = HanaDB._sanitize_list_float(embedding)
    distance_func_name = HANA_DISTANCE_FUNCTION[self.distance_strategy][0]
    embedding_as_str = ','.join(map(str, embedding))
    sql_str = f'SELECT TOP {k}  "{self.content_column}",   "{self.metadata_column}",   TO_NVARCHAR("{self.vector_column}"),   {distance_func_name}("{self.vector_column}", TO_REAL_VECTOR      (ARRAY({embedding_as_str}))) AS CS FROM "{self.table_name}"'
    order_str = f' order by CS {HANA_DISTANCE_FUNCTION[self.distance_strategy][1]}'
    where_str, query_tuple = self._create_where_by_filter(filter)
    sql_str = sql_str + where_str
    sql_str = sql_str + order_str
    try:
        cur = self.connection.cursor()
        cur.execute(sql_str, query_tuple)
        if cur.has_result_set():
            rows = cur.fetchall()
            for row in rows:
                js = json.loads(row[1])
                doc = Document(page_content=row[0], metadata=js)
                result_vector = HanaDB._parse_float_array_from_string(row[2])
                result.append((doc, row[3], result_vector))
    finally:
        cur.close()
    return result