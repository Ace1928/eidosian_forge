import os
import pprint
from twython import Twython
def credsfromfile(creds_file=None, subdir=None, verbose=False):
    """
    Convenience function for authentication
    """
    return Authenticate().load_creds(creds_file=creds_file, subdir=subdir, verbose=verbose)