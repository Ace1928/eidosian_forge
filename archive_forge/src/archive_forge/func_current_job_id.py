from typing import TYPE_CHECKING
from types import SimpleNamespace
@property
def current_job_id(self) -> 'JobID':
    from ray import JobID
    return JobID(self._fetch_runtime_context().job_id)