import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def create_results_with_score(self, json_result: Dict[str, Any]) -> List[Tuple[Document, float]]:
    """Parsing the returned results with scores.
        Args:
            json_result: Results from OpenSearch query.
        Returns:
            query_result_list: Results with scores.
        """
    items = json_result['result']
    query_result_list: List[Tuple[Document, float]] = []
    for item in items:
        fields = item['fields']
        query_result_list.append((Document(page_content=fields[self.config.field_name_mapping['document']], metadata=self.create_inverse_metadata(fields)), float(item['score'])))
    return query_result_list