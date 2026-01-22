import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
@staticmethod
def from_stream(credential_filename):
    """Create a Credentials object by reading information from a file.

        It returns an object of type GoogleCredentials.

        Args:
            credential_filename: the path to the file from where the
                                 credentials are to be read

        Raises:
            ApplicationDefaultCredentialsError: raised when the credentials
                                                fail to be retrieved.
        """
    if credential_filename and os.path.isfile(credential_filename):
        try:
            return _get_application_default_credential_from_file(credential_filename)
        except (ApplicationDefaultCredentialsError, ValueError) as error:
            extra_help = ' (provided as parameter to the from_stream() method)'
            _raise_exception_for_reading_json(credential_filename, extra_help, error)
    else:
        raise ApplicationDefaultCredentialsError('The parameter passed to the from_stream() method should point to a file.')