from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
def delete_load_balancer_service(self):
    try:
        self._get_load_balancer()
        if self.hcloud_load_balancer_service is not None:
            if not self.module.check_mode:
                try:
                    self.hcloud_load_balancer.delete_service(self.hcloud_load_balancer_service).wait_until_finished(max_retries=1000)
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
            self._mark_as_changed()
        self.hcloud_load_balancer_service = None
    except APIException as exception:
        self.fail_json_hcloud(exception)