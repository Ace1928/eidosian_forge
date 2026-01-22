from typing import Any, Iterator, List, Sequence, cast
from langchain_core.documents import BaseDocumentTransformer, Document
@staticmethod
def remove_unwanted_tags(html_content: str, unwanted_tags: List[str]) -> str:
    """
        Remove unwanted tags from a given HTML content.

        Args:
            html_content: The original HTML content string.
            unwanted_tags: A list of tags to be removed from the HTML.

        Returns:
            A cleaned HTML string with unwanted tags removed.
        """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()
    return str(soup)