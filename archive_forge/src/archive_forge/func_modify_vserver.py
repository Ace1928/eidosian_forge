from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def modify_vserver(svm_cx, module, name, modify, parameters=None):
    """
    Modify vserver.
    :param name: vserver name
    :param modify: list of modify attributes
    :param parameters: customer original inputs
    modify only contains the difference between the customer inputs and current
    for some attributes, it may be safer to apply the original inputs
    """
    if parameters is None:
        parameters = modify
    vserver_modify = netapp_utils.zapi.NaElement('vserver-modify')
    vserver_modify.add_new_child('vserver-name', name)
    for attribute in modify:
        if attribute == 'comment':
            vserver_modify.add_new_child('comment', parameters['comment'])
        if attribute == 'language':
            vserver_modify.add_new_child('language', parameters['language'])
        if attribute == 'quota_policy':
            vserver_modify.add_new_child('quota-policy', parameters['quota_policy'])
        if attribute == 'snapshot_policy':
            vserver_modify.add_new_child('snapshot-policy', parameters['snapshot_policy'])
        if attribute == 'max_volumes':
            vserver_modify.add_new_child('max-volumes', parameters['max_volumes'])
        if attribute == 'allowed_protocols':
            allowed_protocols = netapp_utils.zapi.NaElement('allowed-protocols')
            for protocol in parameters['allowed_protocols']:
                allowed_protocols.add_new_child('protocol', protocol)
            vserver_modify.add_child_elem(allowed_protocols)
        if attribute == 'aggr_list':
            aggregates = netapp_utils.zapi.NaElement('aggr-list')
            for aggr in parameters['aggr_list']:
                aggregates.add_new_child('aggr-name', aggr)
            vserver_modify.add_child_elem(aggregates)
    try:
        svm_cx.invoke_successfully(vserver_modify, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as exc:
        module.fail_json(msg='Error modifying SVM %s: %s' % (name, to_native(exc)), exception=traceback.format_exc())