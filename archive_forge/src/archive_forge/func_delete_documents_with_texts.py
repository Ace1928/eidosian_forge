import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def delete_documents_with_texts(self, texts: List[str]) -> bool:
    """Delete documents based on their page content.

        Args:
            texts: List of document page content.
        Returns:
           Whether the deletion was successful or not.
        """
    id_list = [sha1(t.encode('utf-8')).hexdigest() for t in texts]
    return self.delete_documents_with_document_id(id_list)