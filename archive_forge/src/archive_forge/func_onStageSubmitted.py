from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
def onStageSubmitted(self, stageSubmitted):
    stage_info = stageSubmitted.stageInfo()
    message = 'Stage {} Submitted'.format(stage_info.stageId())
    data = {'attemptId': stage_info.attemptId(), 'name': stage_info.name()}
    self.hub.add_breadcrumb(level='info', message=message, data=data)
    _set_app_properties()