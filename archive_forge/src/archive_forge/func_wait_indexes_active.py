from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
def wait_indexes_active(module, wait_timeout, table_name):
    _do_wait(module, 'global_indexes_active', 'secondary index updates', wait_timeout, table_name)