import base64
def authorizeSession(self, client):
    client.add_credentials(self.username, self.password)