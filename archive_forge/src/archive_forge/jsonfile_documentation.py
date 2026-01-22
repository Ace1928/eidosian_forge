from __future__ import (absolute_import, division, print_function)
import codecs
import json
from ansible.parsing.ajson import AnsibleJSONEncoder, AnsibleJSONDecoder
from ansible.plugins.cache import BaseFileCacheModule

    A caching module backed by json files.
    