from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def get_mongodb_client(module, login_user=None, login_password=None, login_database=None, directConnection=False):
    """
    Build the connection params dict and returns a MongoDB Client object
    """
    connection_params = {'host': module.params['login_host'], 'port': module.params['login_port']}
    if directConnection:
        connection_params['directConnection'] = True
    if module.params['ssl']:
        connection_params = ssl_connection_options(connection_params, module)
        connection_params = rename_ssl_option_for_pymongo4(connection_params)
    if 'replica_set' in module.params and 'reconfigure' not in module.params:
        connection_params['replicaset'] = module.params['replica_set']
    elif 'replica_set' in module.params and 'reconfigure' in module.params and module.params['reconfigure']:
        connection_params['replicaset'] = module.params['replica_set']
    if login_user:
        connection_params['username'] = login_user
        connection_params['password'] = login_password
        connection_params['authSource'] = login_database
    client = MongoClient(**connection_params)
    return client