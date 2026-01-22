from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.hrobot.plugins.module_utils.robot import (
def get_failover_record(module, ip):
    """
    Get information record of failover IP.

    See https://robot.your-server.de/doc/webservice/en.html#get-failover-failover-ip
    """
    url = '{0}/failover/{1}'.format(BASE_URL, ip)
    result, error = fetch_url_json(module, url)
    if 'failover' not in result:
        module.fail_json(msg='Cannot interpret result: {0}'.format(json.dumps(result, sort_keys=True)))
    return result['failover']