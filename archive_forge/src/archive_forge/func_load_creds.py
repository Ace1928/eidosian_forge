import os
import pprint
from twython import Twython
def load_creds(self, creds_file=None, subdir=None, verbose=False):
    """
        Read OAuth credentials from a text file.

        File format for OAuth 1::

           app_key=YOUR_APP_KEY
           app_secret=YOUR_APP_SECRET
           oauth_token=OAUTH_TOKEN
           oauth_token_secret=OAUTH_TOKEN_SECRET


        File format for OAuth 2::

           app_key=YOUR_APP_KEY
           app_secret=YOUR_APP_SECRET
           access_token=ACCESS_TOKEN

        :param str file_name: File containing credentials. ``None`` (default) reads
            data from `TWITTER/'credentials.txt'`
        """
    if creds_file is not None:
        self.creds_file = creds_file
    if subdir is None:
        if self.creds_subdir is None:
            msg = "Supply a value to the 'subdir' parameter or" + ' set the TWITTER environment variable.'
            raise ValueError(msg)
    else:
        self.creds_subdir = subdir
    self.creds_fullpath = os.path.normpath(os.path.join(self.creds_subdir, self.creds_file))
    if not os.path.isfile(self.creds_fullpath):
        raise OSError(f'Cannot find file {self.creds_fullpath}')
    with open(self.creds_fullpath) as infile:
        if verbose:
            print(f'Reading credentials file {self.creds_fullpath}')
        for line in infile:
            if '=' in line:
                name, value = line.split('=', 1)
                self.oauth[name.strip()] = value.strip()
    self._validate_creds_file(verbose=verbose)
    return self.oauth