from __future__ import annotations
import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
def get_relevant_documents_with_filter(self, query: str, *, _filter: Optional[str]=None) -> List[Document]:
    body = self.body.copy()
    _filter = f' and {_filter}' if _filter else ''
    body['yql'] = body['yql'] + _filter
    body['query'] = query
    return self._query(body)