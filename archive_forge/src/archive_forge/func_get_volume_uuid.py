from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_volume_uuid(rest_api, volume_name, svm_name, module):
    api = 'storage/volumes'
    query = {'name': volume_name, 'svm.name': svm_name}
    record, error = rest_generic.get_one_record(rest_api, api, query)
    if error:
        module.fail_json(msg='Could not find volume %s on SVM %s' % (volume_name, svm_name))
    return record['uuid'] if record else None