import uuid
def get_origin_access_identity(self):
    return self.connection.get_origin_access_identity_info(self.id)