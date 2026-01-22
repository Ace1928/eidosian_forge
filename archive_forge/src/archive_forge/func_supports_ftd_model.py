from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
@classmethod
def supports_ftd_model(cls, model):
    return model in cls.PLATFORM_MODELS