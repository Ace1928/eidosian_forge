from keystoneauth1 import session
from heat.common import context
def get_ec2_keypair(self, access, user_id):
    if user_id == self.user_id:
        if access == self.access:
            return self.creds
        else:
            raise ValueError('Unexpected access %s' % access)
    else:
        raise ValueError('Unexpected user_id %s' % user_id)