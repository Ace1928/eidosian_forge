from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
@staticmethod
def load_suggestions(query: str='', doc_type: str='all') -> List[Document]:
    """Load suggestions.

        Args:
            query: A query string
            doc_type: The type of document to search for. Can be one of "all",
              "device", "guide", "teardown", "answer", "wiki".

        Returns:

        """
    res = requests.get(IFIXIT_BASE_URL + '/suggest/' + query + '?doctypes=' + doc_type)
    if res.status_code != 200:
        raise ValueError('Could not load suggestions for "' + query + '"\n' + res.json())
    data = res.json()
    results = data['results']
    output = []
    for result in results:
        try:
            loader = IFixitLoader(result['url'])
            if loader.page_type == 'Device':
                output += loader.load_device(include_guides=False)
            else:
                output += loader.load()
        except ValueError:
            continue
    return output