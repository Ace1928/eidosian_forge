import os
import requests
import testtools
from keystone.tests.common import auth as common_auth
def get_scoped_user_token(self):
    return self.get_scoped_token(self.user)