import shutil
from django.db.backends.base.client import BaseDatabaseClient
@staticmethod
def connect_string(settings_dict):
    from django.db.backends.oracle.utils import dsn
    return '%s/"%s"@%s' % (settings_dict['USER'], settings_dict['PASSWORD'], dsn(settings_dict))