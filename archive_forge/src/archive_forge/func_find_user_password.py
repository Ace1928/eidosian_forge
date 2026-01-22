import getpass
from . import get_password, delete_password, set_password
def find_user_password(self, realm, authuri):
    user = self.get_username(realm, authuri)
    password = get_password(realm, user)
    if password is None:
        prompt = 'password for %(user)s@%(realm)s for %(authuri)s: ' % vars()
        password = getpass.getpass(prompt)
        set_password(realm, user, password)
    return (user, password)