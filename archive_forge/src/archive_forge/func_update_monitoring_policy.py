from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def update_monitoring_policy(module, oneandone_conn):
    """
    Updates a monitoring_policy based on input arguments.
    Monitoring policy ports, processes and servers can be added/removed to/from
    a monitoring policy. Monitoring policy name, description, email,
    thresholds for cpu, ram, disk, transfer and internal_ping
    can be updated as well.

    module : AnsibleModule object
    oneandone_conn: authenticated oneandone object
    """
    try:
        monitoring_policy_id = module.params.get('monitoring_policy')
        name = module.params.get('name')
        description = module.params.get('description')
        email = module.params.get('email')
        thresholds = module.params.get('thresholds')
        add_ports = module.params.get('add_ports')
        update_ports = module.params.get('update_ports')
        remove_ports = module.params.get('remove_ports')
        add_processes = module.params.get('add_processes')
        update_processes = module.params.get('update_processes')
        remove_processes = module.params.get('remove_processes')
        add_servers = module.params.get('add_servers')
        remove_servers = module.params.get('remove_servers')
        changed = False
        monitoring_policy = get_monitoring_policy(oneandone_conn, monitoring_policy_id, True)
        if monitoring_policy is None:
            _check_mode(module, False)
        _monitoring_policy = oneandone.client.MonitoringPolicy(name=name, description=description, email=email)
        _thresholds = None
        if thresholds:
            threshold_entities = ['cpu', 'ram', 'disk', 'internal_ping', 'transfer']
            _thresholds = []
            for threshold in thresholds:
                key = list(threshold.keys())[0]
                if key in threshold_entities:
                    _threshold = oneandone.client.Threshold(entity=key, warning_value=threshold[key]['warning']['value'], warning_alert=str(threshold[key]['warning']['alert']).lower(), critical_value=threshold[key]['critical']['value'], critical_alert=str(threshold[key]['critical']['alert']).lower())
                    _thresholds.append(_threshold)
        if name or description or email or thresholds:
            _check_mode(module, True)
            monitoring_policy = oneandone_conn.modify_monitoring_policy(monitoring_policy_id=monitoring_policy['id'], monitoring_policy=_monitoring_policy, thresholds=_thresholds)
            changed = True
        if add_ports:
            if module.check_mode:
                _check_mode(module, _add_ports(module, oneandone_conn, monitoring_policy['id'], add_ports))
            monitoring_policy = _add_ports(module, oneandone_conn, monitoring_policy['id'], add_ports)
            changed = True
        if update_ports:
            chk_changed = False
            for update_port in update_ports:
                if module.check_mode:
                    chk_changed |= _modify_port(module, oneandone_conn, monitoring_policy['id'], update_port['id'], update_port)
                _modify_port(module, oneandone_conn, monitoring_policy['id'], update_port['id'], update_port)
            monitoring_policy = get_monitoring_policy(oneandone_conn, monitoring_policy['id'], True)
            changed = True
        if remove_ports:
            chk_changed = False
            for port_id in remove_ports:
                if module.check_mode:
                    chk_changed |= _delete_monitoring_policy_port(module, oneandone_conn, monitoring_policy['id'], port_id)
                _delete_monitoring_policy_port(module, oneandone_conn, monitoring_policy['id'], port_id)
            _check_mode(module, chk_changed)
            monitoring_policy = get_monitoring_policy(oneandone_conn, monitoring_policy['id'], True)
            changed = True
        if add_processes:
            monitoring_policy = _add_processes(module, oneandone_conn, monitoring_policy['id'], add_processes)
            _check_mode(module, monitoring_policy)
            changed = True
        if update_processes:
            chk_changed = False
            for update_process in update_processes:
                if module.check_mode:
                    chk_changed |= _modify_process(module, oneandone_conn, monitoring_policy['id'], update_process['id'], update_process)
                _modify_process(module, oneandone_conn, monitoring_policy['id'], update_process['id'], update_process)
            _check_mode(module, chk_changed)
            monitoring_policy = get_monitoring_policy(oneandone_conn, monitoring_policy['id'], True)
            changed = True
        if remove_processes:
            chk_changed = False
            for process_id in remove_processes:
                if module.check_mode:
                    chk_changed |= _delete_monitoring_policy_process(module, oneandone_conn, monitoring_policy['id'], process_id)
                _delete_monitoring_policy_process(module, oneandone_conn, monitoring_policy['id'], process_id)
            _check_mode(module, chk_changed)
            monitoring_policy = get_monitoring_policy(oneandone_conn, monitoring_policy['id'], True)
            changed = True
        if add_servers:
            monitoring_policy = _attach_monitoring_policy_server(module, oneandone_conn, monitoring_policy['id'], add_servers)
            _check_mode(module, monitoring_policy)
            changed = True
        if remove_servers:
            chk_changed = False
            for _server_id in remove_servers:
                server_id = get_server(oneandone_conn, _server_id)
                if module.check_mode:
                    chk_changed |= _detach_monitoring_policy_server(module, oneandone_conn, monitoring_policy['id'], server_id)
                _detach_monitoring_policy_server(module, oneandone_conn, monitoring_policy['id'], server_id)
            _check_mode(module, chk_changed)
            monitoring_policy = get_monitoring_policy(oneandone_conn, monitoring_policy['id'], True)
            changed = True
        return (changed, monitoring_policy)
    except Exception as ex:
        module.fail_json(msg=str(ex))