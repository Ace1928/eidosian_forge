from typing import Any, Iterator, List, Sequence, cast
from langchain_core.documents import BaseDocumentTransformer, Document
def transform_documents(self, documents: Sequence[Document], unwanted_tags: List[str]=['script', 'style'], tags_to_extract: List[str]=['p', 'li', 'div', 'a'], remove_lines: bool=True, *, remove_comments: bool=False, **kwargs: Any) -> Sequence[Document]:
    """
        Transform a list of Document objects by cleaning their HTML content.

        Args:
            documents: A sequence of Document objects containing HTML content.
            unwanted_tags: A list of tags to be removed from the HTML.
            tags_to_extract: A list of tags whose content will be extracted.
            remove_lines: If set to True, unnecessary lines will be removed.
            remove_comments: If set to True, comments will be removed.

        Returns:
            A sequence of Document objects with transformed content.
        """
    for doc in documents:
        cleaned_content = doc.page_content
        cleaned_content = self.remove_unwanted_tags(cleaned_content, unwanted_tags)
        cleaned_content = self.extract_tags(cleaned_content, tags_to_extract, remove_comments=remove_comments)
        if remove_lines:
            cleaned_content = self.remove_unnecessary_lines(cleaned_content)
        doc.page_content = cleaned_content
    return documents