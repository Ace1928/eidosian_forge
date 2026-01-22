from typing import Any, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
Embed a query using the text2vec embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        