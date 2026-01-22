from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def modify_quota_config(self, quota_config_obj, quota_config_params):
    """
        Modify default quota config settings of newly created filesystem.
        The default setting of quota config after filesystem creation is:
        default_soft_limit and default_hard_limit are 0,
        is_user_quota_enabled is false,
        grace_period is 7 days and,
        quota_policy is FILE_SIZE.
        :param quota_config_obj: Quota config instance
        :param quota_config_params: Quota config parameters to be modified
        :return: Boolean whether quota config is modified
        """
    if quota_config_params:
        soft_limit = quota_config_params['default_soft_limit']
        hard_limit = quota_config_params['default_hard_limit']
        is_user_quota_enabled = quota_config_params['is_user_quota_enabled']
        quota_policy = quota_config_params['quota_policy']
        grace_period = quota_config_params['grace_period']
        cap_unit = quota_config_params['cap_unit']
        gp_unit = quota_config_params['grace_period_unit']
    if soft_limit:
        soft_limit_in_bytes = utils.get_size_bytes(soft_limit, cap_unit)
    else:
        soft_limit_in_bytes = quota_config_obj.default_soft_limit
    if hard_limit:
        hard_limit_in_bytes = utils.get_size_bytes(hard_limit, cap_unit)
    else:
        hard_limit_in_bytes = quota_config_obj.default_hard_limit
    if grace_period:
        grace_period_in_sec = get_time_in_seconds(grace_period, gp_unit)
    else:
        grace_period_in_sec = quota_config_obj.grace_period
    policy_enum = None
    policy_enum_val = None
    if quota_policy:
        if utils.QuotaPolicyEnum[quota_policy]:
            policy_enum = utils.QuotaPolicyEnum[quota_policy]
            policy_enum_val = utils.QuotaPolicyEnum[quota_policy]._get_properties()['value']
        else:
            errormsg = 'Invalid choice {0} for quota policy'.format(quota_policy)
            LOG.error(errormsg)
            self.module.fail_json(msg=errormsg)
    if quota_config_obj.default_hard_limit == hard_limit_in_bytes and quota_config_obj.default_soft_limit == soft_limit_in_bytes and (quota_config_obj.grace_period == grace_period_in_sec) and (quota_policy is not None and quota_config_obj.quota_policy == policy_enum or quota_policy is None) and (is_user_quota_enabled is None or (is_user_quota_enabled is not None and is_user_quota_enabled == quota_config_obj.is_user_quota_enabled)):
        return False
    try:
        resp = self.unity_conn.modify_quota_config(quota_config_id=quota_config_obj.id, grace_period=grace_period_in_sec, default_hard_limit=hard_limit_in_bytes, default_soft_limit=soft_limit_in_bytes, is_user_quota_enabled=is_user_quota_enabled, quota_policy=policy_enum_val)
        LOG.info('Successfully modified the quota config with response %s', resp)
        return True
    except Exception as e:
        errormsg = 'Failed to modify quota config for filesystem {0}  with error {1}'.format(quota_config_obj.filesystem.id, str(e))
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)