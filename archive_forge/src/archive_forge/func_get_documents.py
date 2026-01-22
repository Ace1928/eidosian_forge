from __future__ import annotations
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
def get_documents(self, ids: Optional[List[str]]=None, filter: Optional[Dict[str, Any]]=None) -> List[Document]:
    """Search documents by their ids or metadata values.

        Args:
            ids: List of ids of documents to retrieve from the vectorstore.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
    if ids and len(ids) > 0:
        from google.cloud import bigquery
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter('ids', 'STRING', ids)])
        id_expr = f'{self.doc_id_field} IN UNNEST(@ids)'
    else:
        job_config = None
        id_expr = 'TRUE'
    if filter:
        filter_expressions = []
        for i in filter.items():
            if isinstance(i[1], float):
                expr = f"ABS(CAST(JSON_VALUE(`{self.metadata_field}`,'$.{i[0]}') AS FLOAT64) - {i[1]}) <= {sys.float_info.epsilon}"
            else:
                val = str(i[1]).replace('"', '\\"')
                expr = f'''JSON_VALUE(`{self.metadata_field}`,'$.{i[0]}') = "{val}"'''
            filter_expressions.append(expr)
        filter_expression_str = ' AND '.join(filter_expressions)
        where_filter_expr = f' AND ({filter_expression_str})'
    else:
        where_filter_expr = ''
    job = self.bq_client.query(f'\n                    SELECT * FROM `{self.full_table_id}` WHERE {id_expr}\n                    {where_filter_expr}\n                    ', job_config=job_config)
    docs: List[Document] = []
    for row in job:
        metadata = None
        if self.metadata_field:
            metadata = row[self.metadata_field]
        if metadata:
            if not isinstance(metadata, dict):
                metadata = json.loads(metadata)
        else:
            metadata = {}
        metadata['__id'] = row[self.doc_id_field]
        doc = Document(page_content=row[self.content_field], metadata=metadata)
        docs.append(doc)
    return docs