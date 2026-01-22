import hashlib
import hmac
import os
import struct
from ntlm_auth.compute_response import ComputeResponse
from ntlm_auth.constants import AvId, AvFlags, MessageTypes, NegotiateFlags, \
from ntlm_auth.rc4 import ARC4
def get_random_export_session_key():
    return os.urandom(16)