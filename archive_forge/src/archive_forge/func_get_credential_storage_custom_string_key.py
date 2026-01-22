import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
@util.positional(2)
def get_credential_storage_custom_string_key(filename, key_string, warn_on_readonly=True):
    """Get a Storage instance for a credential using a single string as a key.

    Allows you to provide a string as a custom key that will be used for
    credential storage and retrieval.

    Args:
        filename: The JSON file storing a set of credentials
        key_string: A string to use as the key for storing this credential.
        warn_on_readonly: if True, log a warning if the store is readonly

    Returns:
        An object derived from client.Storage for getting/setting the
        credential.
    """
    key_dict = {'key': key_string}
    return get_credential_storage_custom_key(filename, key_dict, warn_on_readonly=warn_on_readonly)