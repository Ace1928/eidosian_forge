import logging
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def lazy_load(self, query: str) -> Iterator[Document]:
    """
        Run Wikipedia search and get the article text plus the meta information.
        See

        Returns: a list of documents.

        """
    page_titles = self.wiki_client.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=self.top_k_results)
    for page_title in page_titles[:self.top_k_results]:
        if (wiki_page := self._fetch_page(page_title)):
            if (doc := self._page_to_document(page_title, wiki_page)):
                yield doc