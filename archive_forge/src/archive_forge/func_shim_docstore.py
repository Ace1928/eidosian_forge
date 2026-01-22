from enum import Enum
from typing import Dict, List, Optional
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.vectorstores import VectorStore
from langchain.storage._lc_store import create_kv_docstore
@root_validator(pre=True)
def shim_docstore(cls, values: Dict) -> Dict:
    byte_store = values.get('byte_store')
    docstore = values.get('docstore')
    if byte_store is not None:
        docstore = create_kv_docstore(byte_store)
    elif docstore is None:
        raise Exception('You must pass a `byte_store` parameter.')
    values['docstore'] = docstore
    return values