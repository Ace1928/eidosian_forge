from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_images_common import (
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def delete_from_registry(self, url):
    try:
        response = self.rest_client.DELETE(url=url, headers=self.client.configuration.api_key)
        if response.status == 404:
            return None
        if response.status < 200 or response.status >= 400:
            return None
        if response.status != 202 and response.status != 204:
            self.fail_json(msg='Delete URL {0}: Unexpected status code in response: {1}'.format(response.status, url), reason=response.reason)
        return None
    except ApiException as e:
        if e.status != 404:
            self.fail_json(msg='Failed to delete URL: %s' % url, reason=e.reason, status=e.status)
    except Exception as e:
        self.fail_json(msg='Delete URL {0}: {1}'.format(url, type(e)))