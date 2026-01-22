import os
import platform
import socket
import stat
import six
from oauthlib import oauth1
from six.moves.urllib.parse import parse_qs, urlencode
from lazr.restfulclient.authorize import HttpAuthorizer
from lazr.restfulclient.errors import CredentialsFileError
def save_to_path(self, path):
    """Convenience method for saving credentials to a file.

        Create the file, call self.save(), and close the
        file. Existing files are overwritten. The resulting file will
        be readable and writable only by the user.

        :param path: In which file the credential file should be saved.
        :type path: string
        """
    credentials_file = os.fdopen(os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, stat.S_IREAD | stat.S_IWRITE), 'w')
    self.save(credentials_file)
    credentials_file.close()