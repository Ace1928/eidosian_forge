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
def batch_parse(self, blobs: Sequence[Blob], gcs_output_path: Optional[str]=None, timeout_sec: int=3600, check_in_interval_sec: int=60) -> Iterator[Document]:
    """Parses a list of blobs lazily.

        Args:
            blobs: a list of blobs to parse.
            gcs_output_path: a path on Google Cloud Storage to store parsing results.
            timeout_sec: a timeout to wait for Document AI to complete, in seconds.
            check_in_interval_sec: an interval to wait until next check
                whether parsing operations have been completed, in seconds
        This is a long-running operation. A recommended way is to decouple
            parsing from creating LangChain Documents:
            >>> operations = parser.docai_parse(blobs, gcs_path)
            >>> parser.is_running(operations)
            You can get operations names and save them:
            >>> names = [op.operation.name for op in operations]
            And when all operations are finished, you can use their results:
            >>> operations = parser.operations_from_names(operation_names)
            >>> results = parser.get_results(operations)
            >>> docs = parser.parse_from_results(results)
        """
    output_path = gcs_output_path or self._gcs_output_path
    if not output_path:
        raise ValueError('An output path on Google Cloud Storage should be provided.')
    operations = self.docai_parse(blobs, gcs_output_path=output_path)
    operation_names = [op.operation.name for op in operations]
    logger.debug('Started parsing with Document AI, submitted operations %s', operation_names)
    time_elapsed = 0
    while self.is_running(operations):
        time.sleep(check_in_interval_sec)
        time_elapsed += check_in_interval_sec
        if time_elapsed > timeout_sec:
            raise TimeoutError(f'Timeout exceeded! Check operations {operation_names} later!')
        logger.debug('.')
    results = self.get_results(operations=operations)
    yield from self.parse_from_results(results)