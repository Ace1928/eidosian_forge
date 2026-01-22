from typing import Callable, Optional
from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore, ByteStore
from langchain.storage.encoder_backed import EncoderBackedStore
def create_lc_store(store: ByteStore, *, key_encoder: Optional[Callable[[str], str]]=None) -> BaseStore[str, Serializable]:
    """Create a store for langchain serializable objects from a bytes store.

    Args:
        store: A bytes store to use as the underlying store.
        key_encoder: A function to encode keys; if None uses identity function.

    Returns:
        A key-value store for documents.
    """
    return EncoderBackedStore(store, key_encoder or _identity, _dump_as_bytes, _load_from_bytes)