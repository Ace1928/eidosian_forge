from __future__ import (absolute_import, division, print_function)
import json
import zipfile
import io
def is_standalone(self):
    headers = {'Content-Type': 'application/json'}
    url = 'https://{ip}/api/v1/deployment/node/{hostname}'.format(ip=self.ip, hostname=self.hostname)
    response = False
    try:
        response = requests.get(url=url, headers=headers, auth=(self.username, self.password), verify=False)
    except Exception as e:
        raise AnsibleActionFail("Couldn't connect, the node might be still initializing, try again in a few minutes. Error received: {e}".format(e=e))
    if not response:
        raise AnsibleActionFail("Couldn't get a valid response from the API. Maybe the node is still initializing, try again in a few minutes.")
    else:
        response = json.loads(response.text).get('response')
        if 'Standalone' in response.get('roles'):
            return True
    return False