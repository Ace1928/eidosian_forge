import threading
import keyring
from oauth2client import client
def locked_get(self):
    """Retrieve Credential from file.

        Returns:
            oauth2client.client.Credentials
        """
    credentials = None
    content = keyring.get_password(self._service_name, self._user_name)
    if content is not None:
        try:
            credentials = client.Credentials.new_from_json(content)
            credentials.set_store(self)
        except ValueError:
            pass
    return credentials