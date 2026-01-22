from typing import Any, Sequence
from langchain_core.documents import BaseDocumentTransformer, Document
Replace occurrences of a particular search pattern with a replacement string

    Arguments:
        ignore_links: Whether links should be ignored; defaults to True.
        ignore_images: Whether images should be ignored; defaults to True.

    Example:
        .. code-block:: python
            from langchain_community.document_transformers import Html2TextTransformer
            html2text = Html2TextTransformer()
            docs_transform = html2text.transform_documents(docs)
    