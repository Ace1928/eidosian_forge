from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
def parse_pagination_link(header):
    if not re.match(R_LINK_HEADER, header, re.VERBOSE):
        raise ScalewayException('Scaleway API answered with an invalid Link pagination header')
    else:
        relations = header.split(',')
        parsed_relations = {}
        rc_relation = re.compile(R_RELATION)
        for relation in relations:
            match = rc_relation.match(relation)
            if not match:
                raise ScalewayException('Scaleway API answered with an invalid relation in the Link pagination header')
            data = match.groupdict()
            parsed_relations[data['relation']] = data['target_IRI']
        return parsed_relations