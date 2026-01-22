from __future__ import absolute_import, division, print_function
import abc
import datetime
import errno
import hashlib
import os
import re
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from .basic import (
def get_fingerprint_of_bytes(source, prefer_one=False):
    """Generate the fingerprint of the given bytes."""
    fingerprint = {}
    try:
        algorithms = hashlib.algorithms
    except AttributeError:
        try:
            algorithms = hashlib.algorithms_guaranteed
        except AttributeError:
            return None
    if prefer_one:
        prefered_algorithms = [algorithm for algorithm in PREFERRED_FINGERPRINTS if algorithm in algorithms]
        prefered_algorithms += sorted([algorithm for algorithm in algorithms if algorithm not in PREFERRED_FINGERPRINTS])
        algorithms = prefered_algorithms
    for algo in algorithms:
        f = getattr(hashlib, algo)
        try:
            h = f(source)
        except ValueError:
            continue
        try:
            pubkey_digest = h.hexdigest()
        except TypeError:
            pubkey_digest = h.hexdigest(32)
        fingerprint[algo] = ':'.join((pubkey_digest[i:i + 2] for i in range(0, len(pubkey_digest), 2)))
        if prefer_one:
            break
    return fingerprint