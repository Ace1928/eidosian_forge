from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def connect_to_db(module, conn_params, autocommit=False, fail_on_conn=True):
    """Connect to a PostgreSQL database.

    Return a tuple containing a psycopg connection object and error message / None.

    Args:
        module (AnsibleModule) -- object of ansible.module_utils.basic.AnsibleModule class
        conn_params (dict) -- dictionary with connection parameters

    Kwargs:
        autocommit (bool) -- commit automatically (default False)
        fail_on_conn (bool) -- fail if connection failed or just warn and return None (default True)
    """
    db_connection = None
    conn_err = None
    try:
        if PSYCOPG_VERSION >= LooseVersion('3.0'):
            conn_params['autocommit'] = autocommit
            conn_params['cursor_factory'] = ClientCursor
            conn_params['row_factory'] = dict_row
            db_connection = psycopg.connect(**conn_params)
        else:
            db_connection = psycopg2.connect(**conn_params)
            if autocommit:
                if PSYCOPG_VERSION >= LooseVersion('2.4.2'):
                    db_connection.set_session(autocommit=True)
                else:
                    db_connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        if module.params.get('session_role'):
            if PSYCOPG_VERSION >= LooseVersion('3.0'):
                cursor = db_connection.cursor(row_factory=psycopg.rows.dict_row)
            else:
                cursor = db_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            try:
                cursor.execute('SET ROLE "%s"' % module.params['session_role'])
            except Exception as e:
                module.fail_json(msg='Could not switch role: %s' % to_native(e))
            finally:
                cursor.close()
    except TypeError as e:
        if 'sslrootcert' in e.args[0]:
            module.fail_json(msg='Postgresql server must be at least version 8.4 to support sslrootcert')
        conn_err = to_native(e)
    except Exception as e:
        conn_err = to_native(e)
    if conn_err is not None:
        if fail_on_conn:
            module.fail_json(msg='unable to connect to database: %s' % conn_err)
        else:
            module.warn('PostgreSQL server is unavailable: %s' % conn_err)
            db_connection = None
    return (db_connection, conn_err)