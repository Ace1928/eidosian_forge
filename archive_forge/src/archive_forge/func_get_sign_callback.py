import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
import six
from google.auth import exceptions
def get_sign_callback(signer_lib, config_file_path):

    def sign_callback(sig, sig_len, tbs, tbs_len):
        _LOGGER.debug('calling sign callback...')
        digest = _compute_sha256_digest(tbs, tbs_len)
        digestArray = ctypes.c_char * len(digest)
        sig_holder_len = 2000
        sig_holder = ctypes.create_string_buffer(sig_holder_len)
        signature_len = signer_lib.SignForPython(config_file_path.encode(), digestArray.from_buffer(bytearray(digest)), len(digest), sig_holder, sig_holder_len)
        if signature_len == 0:
            return 0
        sig_len[0] = signature_len
        bs = bytearray(sig_holder)
        for i in range(signature_len):
            sig[i] = bs[i]
        return 1
    return SIGN_CALLBACK_CTYPE(sign_callback)