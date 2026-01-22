from __future__ import absolute_import, division, print_function
from ansible.errors import (
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.collections import is_sequence
def hashids_decode(hashid, salt=None, alphabet=None, min_length=None):
    """Decodes a YouTube-like hash to a sequence of ints

       :hashid: Hash string to decode
       :salt: String to use as salt when hashing
       :alphabet: String of 16 or more unique characters to produce a hash
       :min_length: Minimum length of hash produced
    """
    hashids = initialize_hashids(salt=salt, alphabet=alphabet, min_length=min_length)
    nums = hashids.decode(hashid)
    return list(nums)