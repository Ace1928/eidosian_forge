import platform
from google.auth import version
def mds_ping():
    return '{} {}'.format(python_and_auth_lib_version(), REQUEST_TYPE_MDS_PING)