from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_instances_1234_configuration(self, **kw):
    return (200, {}, {'instance': {'configuration': {'tmp_table_size': '15M', 'innodb_log_files_in_group': '2', 'skip-external-locking': '1', 'max_user_connections': '98'}}})