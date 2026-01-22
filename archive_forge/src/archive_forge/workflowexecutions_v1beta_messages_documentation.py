from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
Optional. A view defining which fields should be filled in the
    returned executions. The API will default to the BASIC view.

    Values:
      EXECUTION_VIEW_UNSPECIFIED: The default / unset value.
      BASIC: Includes only basic metadata about the execution. Following
        fields are returned: name, start_time, end_time, state and
        workflow_revision_id.
      FULL: Includes all data.
    