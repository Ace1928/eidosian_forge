import warnings
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.milvus import Milvus
@root_validator(pre=True)
def create_retriever(cls, values: Dict) -> Dict:
    """Create the Milvus store and retriever."""
    values['store'] = Milvus(values['embedding_function'], values['collection_name'], values['collection_properties'], values['connection_args'], values['consistency_level'])
    values['retriever'] = values['store'].as_retriever(search_kwargs={'param': values['search_params']})
    return values