import platform
from google.auth import version
def token_request_id_token_impersonate():
    return '{} {} {}'.format(python_and_auth_lib_version(), REQUEST_TYPE_ID_TOKEN, CRED_TYPE_SA_IMPERSONATE)