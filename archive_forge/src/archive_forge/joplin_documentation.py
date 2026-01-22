import json
import urllib
from datetime import datetime
from typing import Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from langchain_community.document_loaders.base import BaseLoader


        Args:
            access_token: The access token to use.
            port: The port where the Web Clipper service is running. Default is 41184.
            host: The host where the Web Clipper service is running.
                Default is localhost.
        