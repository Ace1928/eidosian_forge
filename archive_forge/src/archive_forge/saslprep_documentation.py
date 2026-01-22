from __future__ import (absolute_import, division, print_function)
from stringprep import (
from unicodedata import normalize
from ansible.module_utils.six import text_type
RFC4013 implementation.
    Implements "SASLprep" profile (RFC4013) of the "stringprep" algorithm (RFC3454)
    to prepare Unicode strings representing user names and passwords for comparison.
    Regarding the RFC4013, the "SASLprep" profile is intended to be used by
    Simple Authentication and Security Layer (SASL) mechanisms
    (such as PLAIN, CRAM-MD5, and DIGEST-MD5), as well as other protocols
    exchanging simple user names and/or passwords.

    Args:
        string (unicode string): Unicode string to validate and prepare.

    Returns:
        Prepared unicode string.
    