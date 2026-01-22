from functools import wraps
import hashlib
import json
import os
import pickle
import six.moves.http_client as httplib
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
def init_app(self, app, scopes=None, client_secrets_file=None, client_id=None, client_secret=None, authorize_callback=None, storage=None, **kwargs):
    """Initialize this extension for the given app.

        Arguments:
            app: A Flask application.
            scopes: Optional list of scopes to authorize.
            client_secrets_file: Path to a file containing client secrets. You
                can also specify the GOOGLE_OAUTH2_CLIENT_SECRETS_FILE config
                value.
            client_id: If not specifying a client secrets file, specify the
                OAuth2 client id. You can also specify the
                GOOGLE_OAUTH2_CLIENT_ID config value. You must also provide a
                client secret.
            client_secret: The OAuth2 client secret. You can also specify the
                GOOGLE_OAUTH2_CLIENT_SECRET config value.
            authorize_callback: A function that is executed after successful
                user authorization.
            storage: A oauth2client.client.Storage subclass for storing the
                credentials. By default, this is a Flask session based storage.
            kwargs: Any additional args are passed along to the Flow
                constructor.
        """
    self.app = app
    self.authorize_callback = authorize_callback
    self.flow_kwargs = kwargs
    if storage is None:
        storage = dictionary_storage.DictionaryStorage(session, key=_CREDENTIALS_KEY)
    self.storage = storage
    if scopes is None:
        scopes = app.config.get('GOOGLE_OAUTH2_SCOPES', _DEFAULT_SCOPES)
    self.scopes = scopes
    self._load_config(client_secrets_file, client_id, client_secret)
    app.register_blueprint(self._create_blueprint())