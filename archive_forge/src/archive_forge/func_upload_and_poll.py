from __future__ import annotations
import asyncio
from typing import List, Iterable
from typing_extensions import Literal
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import httpx
import sniffio
from .... import _legacy_response
from ....types import FileObject
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from ...._utils import (
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ....pagination import SyncCursorPage, AsyncCursorPage
from ...._base_client import (
from ....types.beta.vector_stores import (
def upload_and_poll(self, vector_store_id: str, *, files: Iterable[FileTypes], max_concurrency: int=5, file_ids: List[str]=[], poll_interval_ms: int | NotGiven=NOT_GIVEN) -> VectorStoreFileBatch:
    """Uploads the given files concurrently and then creates a vector store file batch.

        If you've already uploaded certain files that you want to include in this batch
        then you can pass their IDs through the `file_ids` argument.

        By default, if any file upload fails then an exception will be eagerly raised.

        The number of concurrency uploads is configurable using the `max_concurrency`
        parameter.

        Note: this method only supports `asyncio` or `trio` as the backing async
        runtime.
        """
    results: list[FileObject] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: list[Future[FileObject]] = [executor.submit(self._client.files.create, file=file, purpose='assistants') for file in files]
    for future in as_completed(futures):
        exc = future.exception()
        if exc:
            raise exc
        results.append(future.result())
    batch = self.create_and_poll(vector_store_id=vector_store_id, file_ids=[*file_ids, *(f.id for f in results)], poll_interval_ms=poll_interval_ms)
    return batch