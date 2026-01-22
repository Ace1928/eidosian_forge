from __future__ import annotations
import time
import typing_extensions
from typing import Mapping, cast
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import FileObject, FileDeleted, file_list_params, file_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from .._utils import (
from .._compat import cached_property
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import (
from ..pagination import SyncPage, AsyncPage
from .._base_client import (
@typing_extensions.deprecated('The `.content()` method should be used instead')
def retrieve_content(self, file_id: str, *, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> str:
    """
        Returns the contents of the specified file.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
    if not file_id:
        raise ValueError(f'Expected a non-empty value for `file_id` but received {file_id!r}')
    return self._get(f'/files/{file_id}/content', options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=str)