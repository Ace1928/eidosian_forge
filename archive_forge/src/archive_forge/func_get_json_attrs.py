import uuid
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
def get_json_attrs(self, with_user_info=True):
    attrs = {}
    if self.release is not None:
        attrs['release'] = self.release
    if self.environment is not None:
        attrs['environment'] = self.environment
    if with_user_info:
        if self.ip_address is not None:
            attrs['ip_address'] = self.ip_address
        if self.user_agent is not None:
            attrs['user_agent'] = self.user_agent
    return attrs