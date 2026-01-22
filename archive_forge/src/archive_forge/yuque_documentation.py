import re
from typing import Dict, Iterator, List
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Load documents from `Yuque`.