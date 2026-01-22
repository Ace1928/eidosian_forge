import os
import re
from typing import Iterator, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def getEthBalance(self) -> List[Document]:
    url = f'https://api.etherscan.io/api?module=account&action=balance&address={self.account_address}&tag=latest&apikey={self.api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print('Error occurred while making the request:', e)
    return [Document(page_content=response.json()['result'])]