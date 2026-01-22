from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
def wait_table_not_exists(module, wait_timeout, table_name):
    _do_wait(module, 'table_not_exists', 'table deletion', wait_timeout, table_name)