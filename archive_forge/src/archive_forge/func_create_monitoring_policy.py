from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def create_monitoring_policy(module, oneandone_conn):
    """
    Creates a new monitoring policy.

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object
    """
    try:
        name = module.params.get('name')
        description = module.params.get('description')
        email = module.params.get('email')
        agent = module.params.get('agent')
        thresholds = module.params.get('thresholds')
        ports = module.params.get('ports')
        processes = module.params.get('processes')
        wait = module.params.get('wait')
        wait_timeout = module.params.get('wait_timeout')
        wait_interval = module.params.get('wait_interval')
        _monitoring_policy = oneandone.client.MonitoringPolicy(name, description, email, agent)
        _monitoring_policy.specs['agent'] = str(_monitoring_policy.specs['agent']).lower()
        threshold_entities = ['cpu', 'ram', 'disk', 'internal_ping', 'transfer']
        _thresholds = []
        for threshold in thresholds:
            key = list(threshold.keys())[0]
            if key in threshold_entities:
                _threshold = oneandone.client.Threshold(entity=key, warning_value=threshold[key]['warning']['value'], warning_alert=str(threshold[key]['warning']['alert']).lower(), critical_value=threshold[key]['critical']['value'], critical_alert=str(threshold[key]['critical']['alert']).lower())
                _thresholds.append(_threshold)
        _ports = []
        for port in ports:
            _port = oneandone.client.Port(protocol=port['protocol'], port=port['port'], alert_if=port['alert_if'], email_notification=str(port['email_notification']).lower())
            _ports.append(_port)
        _processes = []
        for process in processes:
            _process = oneandone.client.Process(process=process['process'], alert_if=process['alert_if'], email_notification=str(process['email_notification']).lower())
            _processes.append(_process)
        _check_mode(module, True)
        monitoring_policy = oneandone_conn.create_monitoring_policy(monitoring_policy=_monitoring_policy, thresholds=_thresholds, ports=_ports, processes=_processes)
        if wait:
            wait_for_resource_creation_completion(oneandone_conn, OneAndOneResources.monitoring_policy, monitoring_policy['id'], wait_timeout, wait_interval)
        changed = True if monitoring_policy else False
        _check_mode(module, False)
        return (changed, monitoring_policy)
    except Exception as ex:
        module.fail_json(msg=str(ex))