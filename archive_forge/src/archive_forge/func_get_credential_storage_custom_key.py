import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
@util.positional(2)
def get_credential_storage_custom_key(filename, key_dict, warn_on_readonly=True):
    """Get a Storage instance for a credential using a dictionary as a key.

    Allows you to provide a dictionary as a custom key that will be used for
    credential storage and retrieval.

    Args:
        filename: The JSON file storing a set of credentials
        key_dict: A dictionary to use as the key for storing this credential.
                  There is no ordering of the keys in the dictionary. Logically
                  equivalent dictionaries will produce equivalent storage keys.
        warn_on_readonly: if True, log a warning if the store is readonly

    Returns:
        An object derived from client.Storage for getting/setting the
        credential.
    """
    multistore = _get_multistore(filename, warn_on_readonly=warn_on_readonly)
    key = _dict_to_tuple_key(key_dict)
    return multistore._get_storage(key)