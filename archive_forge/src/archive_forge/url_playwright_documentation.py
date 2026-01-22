import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Load the specified URLs with Playwright and create Documents asynchronously.
        Use this function when in a jupyter notebook environment.

        Returns:
            A list of Document instances with loaded content.
        