import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def create_results(self, json_result: Dict[str, Any]) -> List[Document]:
    """Assemble documents."""
    items = json_result['result']
    query_result_list: List[Document] = []
    for item in items:
        if 'fields' not in item or self.config.field_name_mapping['document'] not in item['fields']:
            query_result_list.append(Document())
        else:
            fields = item['fields']
            query_result_list.append(Document(page_content=fields[self.config.field_name_mapping['document']], metadata=self.create_inverse_metadata(fields)))
    return query_result_list