from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def load_guide(self, url_override: Optional[str]=None) -> List[Document]:
    """Load a guide

        Args:
            url_override: A URL to override the default URL.

        Returns: List[Document]

        """
    if url_override is None:
        url = IFIXIT_BASE_URL + '/guides/' + self.id
    else:
        url = url_override
    res = requests.get(url)
    if res.status_code != 200:
        raise ValueError('Could not load guide: ' + self.web_path + '\n' + res.json())
    data = res.json()
    doc_parts = ['# ' + data['title'], data['introduction_raw']]
    doc_parts.append('\n\n###Tools Required:')
    if len(data['tools']) == 0:
        doc_parts.append('\n - None')
    else:
        for tool in data['tools']:
            doc_parts.append('\n - ' + tool['text'])
    doc_parts.append('\n\n###Parts Required:')
    if len(data['parts']) == 0:
        doc_parts.append('\n - None')
    else:
        for part in data['parts']:
            doc_parts.append('\n - ' + part['text'])
    for row in data['steps']:
        doc_parts.append('\n\n## ' + (row['title'] if row['title'] != '' else 'Step {}'.format(row['orderby'])))
        for line in row['lines']:
            doc_parts.append(line['text_raw'])
    doc_parts.append(data['conclusion_raw'])
    text = '\n'.join(doc_parts)
    metadata = {'source': self.web_path, 'title': data['title']}
    return [Document(page_content=text, metadata=metadata)]