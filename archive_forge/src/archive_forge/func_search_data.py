import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def search_data() -> Dict[str, Any]:
    request = QueryRequest(table_name=self.config.table_name, namespace=self.config.namespace, vector=embedding, include_vector=True, output_fields=self.config.output_fields, filter=generate_filter_query(), top_k=k)
    query_result = self.ha3_engine_client.query(request)
    return json.loads(query_result.body)