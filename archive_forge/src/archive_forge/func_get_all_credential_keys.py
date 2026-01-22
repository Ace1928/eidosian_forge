import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
@util.positional(1)
def get_all_credential_keys(filename, warn_on_readonly=True):
    """Gets all the registered credential keys in the given Multistore.

    Args:
        filename: The JSON file storing a set of credentials
        warn_on_readonly: if True, log a warning if the store is readonly

    Returns:
        A list of the credential keys present in the file.  They are returned
        as dictionaries that can be passed into
        get_credential_storage_custom_key to get the actual credentials.
    """
    multistore = _get_multistore(filename, warn_on_readonly=warn_on_readonly)
    multistore._lock()
    try:
        return multistore._get_all_credential_keys()
    finally:
        multistore._unlock()