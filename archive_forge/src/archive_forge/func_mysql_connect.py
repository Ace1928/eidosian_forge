from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.six.moves import configparser
from ansible.module_utils.basic import missing_required_lib
def mysql_connect(module, login_user=None, login_password=None, config_file='', ssl_cert=None, ssl_key=None, ssl_ca=None, db=None, cursor_class=None, connect_timeout=30, autocommit=False, config_overrides_defaults=False):
    config = {}
    if not HAS_MYSQL_PACKAGE:
        module.fail_json(msg=missing_required_lib('pymysql or MySQLdb'), exception=MYSQL_IMP_ERR)
    if module.params['login_port'] < 0 or module.params['login_port'] > 65535:
        module.fail_json(msg='login_port must be a valid unix port number (0-65535)')
    if config_file and os.path.exists(config_file):
        config['read_default_file'] = config_file
        cp = parse_from_mysql_config_file(config_file)
        if cp and cp.has_section('client') and config_overrides_defaults:
            try:
                module.params['login_host'] = cp.get('client', 'host', fallback=module.params['login_host'])
                module.params['login_port'] = cp.getint('client', 'port', fallback=module.params['login_port'])
            except Exception as e:
                if "got an unexpected keyword argument 'fallback'" in e.message:
                    module.fail_json('To use config_overrides_defaults, it needs Python 3.5+ as the default interpreter on a target host')
    if ssl_ca is not None or ssl_key is not None or ssl_cert is not None:
        config['ssl'] = {}
    if module.params['login_unix_socket']:
        config['unix_socket'] = module.params['login_unix_socket']
    else:
        config['host'] = module.params['login_host']
        config['port'] = module.params['login_port']
    if login_user is not None:
        config['user'] = login_user
    if login_password is not None:
        config['passwd'] = login_password
    if ssl_cert is not None:
        config['ssl']['cert'] = ssl_cert
    if ssl_key is not None:
        config['ssl']['key'] = ssl_key
    if ssl_ca is not None:
        config['ssl']['ca'] = ssl_ca
    if db is not None:
        config['db'] = db
    if connect_timeout is not None:
        config['connect_timeout'] = connect_timeout
    if _mysql_cursor_param == 'cursor':
        db_connection = mysql_driver.connect(autocommit=autocommit, **config)
    else:
        db_connection = mysql_driver.connect(**config)
        if autocommit:
            db_connection.autocommit(True)
    version = _version(db_connection.cursor(**{_mysql_cursor_param: mysql_driver.cursors.DictCursor}))
    if cursor_class == 'DictCursor':
        return (db_connection.cursor(**{_mysql_cursor_param: mysql_driver.cursors.DictCursor}), db_connection, version)
    else:
        return (db_connection.cursor(), db_connection, version)