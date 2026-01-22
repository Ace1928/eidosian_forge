from typing import Any, List
from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
def load_results(self, soup: Any) -> List[Document]:
    """Load items from an HN page."""
    items = soup.select("tr[class='athing']")
    documents = []
    for lineItem in items:
        ranking = lineItem.select_one("span[class='rank']").text
        link = lineItem.find('span', {'class': 'titleline'}).find('a').get('href')
        title = lineItem.find('span', {'class': 'titleline'}).text.strip()
        metadata = {'source': self.web_path, 'title': title, 'link': link, 'ranking': ranking}
        documents.append(Document(page_content=title, link=link, ranking=ranking, metadata=metadata))
    return documents