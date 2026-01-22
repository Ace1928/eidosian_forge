from __future__ import annotations
from typing import Union, Optional
from typing_extensions import Literal
import httpx
from ... import _legacy_response
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ..._utils import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ...pagination import SyncCursorPage, AsyncCursorPage
from ..._base_client import (
from ...types.fine_tuning import (
def list_events(self, fine_tuning_job_id: str, *, after: str | NotGiven=NOT_GIVEN, limit: int | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncPaginator[FineTuningJobEvent, AsyncCursorPage[FineTuningJobEvent]]:
    """
        Get status updates for a fine-tuning job.

        Args:
          after: Identifier for the last event from the previous pagination request.

          limit: Number of events to retrieve.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
    if not fine_tuning_job_id:
        raise ValueError(f'Expected a non-empty value for `fine_tuning_job_id` but received {fine_tuning_job_id!r}')
    return self._get_api_list(f'/fine_tuning/jobs/{fine_tuning_job_id}/events', page=AsyncCursorPage[FineTuningJobEvent], options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, query=maybe_transform({'after': after, 'limit': limit}, job_list_events_params.JobListEventsParams)), model=FineTuningJobEvent)