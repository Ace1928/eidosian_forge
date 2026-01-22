from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
def regexp_extraction(string, _regexp, groups=1):
    """ Returns the capture group (default=1) specified in the regexp, applied to the string """
    regexp_search = re.search(string=str(string), pattern=str(_regexp))
    if regexp_search:
        if regexp_search.group(groups) != '':
            return str(regexp_search.group(groups))
    return None