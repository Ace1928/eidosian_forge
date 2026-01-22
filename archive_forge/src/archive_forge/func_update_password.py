from keystoneclient import base
import urllib.parse
def update_password(self, user, password):
    """Update password."""
    params = {'user': {'password': password}}
    return self._update('/users/%s/OS-KSADM/password' % base.getid(user), params, 'user', log=False)