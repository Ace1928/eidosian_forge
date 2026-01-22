import base64
import struct
from ntlm_auth.constants import NegotiateFlags
from ntlm_auth.exceptions import NoAuthContextError
from ntlm_auth.messages import AuthenticateMessage, ChallengeMessage, \
from ntlm_auth.session_security import SessionSecurity
def create_negotiate_message(self, domain_name=None, workstation=None):
    self._context.domain = domain_name
    self._context.workstation = workstation
    msg = self._context.step()
    return base64.b64encode(msg)