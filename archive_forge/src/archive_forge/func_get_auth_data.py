from keystoneauth1.identity.v3 import base
def get_auth_data(self, session, auth, headers, **kwargs):
    user = {'password': self.password}
    if self.user_id:
        user['id'] = self.user_id
    elif self.username:
        user['name'] = self.username
        if self.user_domain_id:
            user['domain'] = {'id': self.user_domain_id}
        elif self.user_domain_name:
            user['domain'] = {'name': self.user_domain_name}
    return ('password', {'user': user})