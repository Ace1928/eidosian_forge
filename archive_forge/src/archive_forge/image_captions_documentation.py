from io import BytesIO
from pathlib import Path
from typing import Any, List, Tuple, Union
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Helper function for getting the captions and metadata of an image.