from keystone.i18n import _
def build_authentication_request(self, token=None, user_id=None, username=None, user_domain_id=None, user_domain_name=None, password=None, kerberos=False, passcode=None, app_cred_id=None, app_cred_name=None, secret=None, **kwargs):
    """Build auth dictionary.

        It will create an auth dictionary based on all the arguments
        that it receives.
        """
    auth_data = {}
    auth_data['identity'] = {'methods': []}
    if kerberos:
        auth_data['identity']['methods'].append('kerberos')
        auth_data['identity']['kerberos'] = {}
    if token:
        auth_data['identity']['methods'].append('token')
        auth_data['identity']['token'] = self._build_token_auth(token)
    if password and (user_id or username):
        auth_data['identity']['methods'].append('password')
        auth_data['identity']['password'] = self._build_auth(user_id, username, user_domain_id, user_domain_name, password=password)
    if passcode and (user_id or username):
        auth_data['identity']['methods'].append('totp')
        auth_data['identity']['totp'] = self._build_auth(user_id, username, user_domain_id, user_domain_name, passcode=passcode)
    if (app_cred_id or app_cred_name) and secret:
        auth_data['identity']['methods'].append('application_credential')
        identity = auth_data['identity']
        identity['application_credential'] = self._build_app_cred_auth(secret, app_cred_id=app_cred_id, app_cred_name=app_cred_name, user_id=user_id, username=username, user_domain_id=user_domain_id, user_domain_name=user_domain_name)
    if kwargs:
        auth_data['scope'] = self._build_auth_scope(**kwargs)
    return {'auth': auth_data}