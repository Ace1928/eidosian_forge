import logging
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import time_util
from saml2.attribute_converter import to_local
from saml2.response import IncorrectlySigned
from saml2.s_utils import OtherError
from saml2.s_utils import VersionMismatch
from saml2.sigver import verify_redirect_signature
from saml2.validate import NotValid
from saml2.validate import valid_instance
def issue_instant_ok(self):
    """Check that the request was issued at a reasonable time"""
    upper = time_util.shift_time(time_util.time_in_a_while(days=1), self.timeslack).timetuple()
    lower = time_util.shift_time(time_util.time_a_while_ago(days=1), -self.timeslack).timetuple()
    issued_at = time_util.str_to_time(self.message.issue_instant)
    return issued_at > lower and issued_at < upper