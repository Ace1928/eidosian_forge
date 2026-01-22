from typing import Iterator, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Initialize with API key, connector id, and account id.

        Args:
            api_key: The Psychic API key.
            account_id: The Psychic account id.
            connector_id: The Psychic connector id.
        