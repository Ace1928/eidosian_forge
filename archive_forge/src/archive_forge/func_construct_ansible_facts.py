from __future__ import (absolute_import, division, print_function)
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGR_RC
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGBaseException
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGRCommon
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import scrub_dict
def construct_ansible_facts(self, response, ansible_params, paramgram, *args, **kwargs):
    """
        Constructs a dictionary to return to ansible facts, containing various information about the execution.

        :param response: Contains the response from the FortiManager.
        :type response: dict
        :param ansible_params: Contains the parameters Ansible was called with.
        :type ansible_params: dict
        :param paramgram: Contains the paramgram passed to the modules' local modify function.
        :type paramgram: dict
        :param args: Free-form arguments that could be added.
        :param kwargs: Free-form keyword arguments that could be added.

        :return: A dictionary containing lots of information to append to Ansible Facts.
        :rtype: dict
        """
    facts = {'response': response, 'ansible_params': scrub_dict(ansible_params), 'paramgram': scrub_dict(paramgram), 'connected_fmgr': self._conn.return_connected_fmgr()}
    if args:
        facts['custom_args'] = args
    if kwargs:
        facts.update(kwargs)
    return facts