import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
@staticmethod
def match_decoders():
    """Return the KVDecoders instance to parse the match section.

        Uses the cached version if available.
        """
    if not ODPFlow._match_decoders:
        ODPFlow._match_decoders = ODPFlow._gen_match_decoders()
    return ODPFlow._match_decoders