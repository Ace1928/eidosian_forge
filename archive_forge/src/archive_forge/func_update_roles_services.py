from __future__ import (absolute_import, division, print_function)
import json
import zipfile
import io
def update_roles_services(self):
    headers = {'Content-Type': 'application/json'}
    url = 'https://{ip}/api/v1/deployment/node/{hostname}'.format(ip=self.ip, hostname=self.hostname)
    data = json.dumps({'roles': self.roles, 'services': self.services})
    try:
        response = requests.put(url=url, timeout=300, auth=(self.username, self.password), headers=headers, data=data, verify=False)
    except Exception as e:
        raise AnsibleActionFail(e)
    if not response:
        raise AnsibleActionFail('Failed to receive a valid response from the API. The actual response was: {response}'.format(response=response.text))