from keystoneauth1 import session
from heat.common import context
def create_stack_user(self, username, password):
    self.username = username
    return self.user_id