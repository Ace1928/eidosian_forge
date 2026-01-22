from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def find_nvdimm_by_label(self, nvdimm_label, nvdimm_devices):
    nvdimm_dev = None
    for nvdimm in nvdimm_devices:
        if nvdimm.deviceInfo.label == nvdimm_label:
            nvdimm_dev = nvdimm
    return nvdimm_dev