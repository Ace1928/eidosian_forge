import asyncio
import json
import uuid
from typing import Any, Sequence
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_community.tools.nuclia.tool import NucliaUnderstandingAPI
Nuclia Text Transformer.

    The Nuclia Understanding API splits into paragraphs and sentences,
    identifies entities, provides a summary of the text and generates
    embeddings for all sentences.
    