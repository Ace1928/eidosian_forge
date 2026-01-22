from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def build_configuration(self):
    """Build audit-log expected configuration.

        :returns: Tuple containing update boolean value and dictionary of audit-log configuration
        """
    config = self.get_configuration()
    current = dict(auditLogMaxRecords=config['auditLogMaxRecords'], auditLogLevel=config['auditLogLevel'], auditLogFullPolicy=config['auditLogFullPolicy'], auditLogWarningThresholdPct=config['auditLogWarningThresholdPct'])
    body = dict(auditLogMaxRecords=self.max_records, auditLogLevel=self.log_level, auditLogFullPolicy=self.full_policy, auditLogWarningThresholdPct=self.threshold)
    update = current != body
    self._logger.info(pformat(update))
    self._logger.info(pformat(body))
    return (update, body)