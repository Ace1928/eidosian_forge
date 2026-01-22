import getpass
from . import get_password, delete_password, set_password
def clear_password(self, realm, authuri):
    user = self.get_username(realm, authuri)
    delete_password(realm, user)