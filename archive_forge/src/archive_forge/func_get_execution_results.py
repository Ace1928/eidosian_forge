from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, cast, Tuple, Union, List, Any
from attr import field
import rpcq
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from pyquil.api import EngagementManager
from pyquil._version import DOCS_URL
def get_execution_results(self, request: GetBuffersRequest) -> GetBuffersResponse:
    """
        Get job buffers and execution metadata.
        """
    result = self._rpcq_request('get_execution_results', job_id=request.job_id, wait=request.wait)
    return GetBuffersResponse(buffers={name: BufferResponse(shape=cast(Tuple[int, int], tuple(val['shape'])), dtype=val['dtype'], data=val['data']) for name, val in result.buffers.items()}, execution_duration_microseconds=result.execution_duration_microseconds)