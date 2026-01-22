import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.utils.iter import batch_iterate
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.utilities.vertexai import get_client_info
def parse_from_results(self, results: List[DocAIParsingResults]) -> Iterator[Document]:
    try:
        from google.cloud.documentai_toolbox.utilities.gcs_utilities import split_gcs_uri
        from google.cloud.documentai_toolbox.wrappers.document import _get_shards
        from google.cloud.documentai_toolbox.wrappers.page import _text_from_layout
    except ImportError as exc:
        raise ImportError('documentai_toolbox package not found, please install it with `pip install google-cloud-documentai-toolbox`') from exc
    for result in results:
        gcs_bucket_name, gcs_prefix = split_gcs_uri(result.parsed_path)
        shards = _get_shards(gcs_bucket_name, gcs_prefix)
        yield from (Document(page_content=_text_from_layout(page.layout, shard.text), metadata={'page': page.page_number, 'source': result.source_path}) for shard in shards for page in shard.pages)