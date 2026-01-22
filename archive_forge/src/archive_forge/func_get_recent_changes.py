import contextlib
import ftplib
import gzip
import os
import re
import shutil
import sys
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.request import urlcleanup
def get_recent_changes(self):
    """Return three lists of the newest weekly files (added,mod,obsolete).

        Reads the directories with changed entries from the PDB server and
        returns a tuple of three URL's to the files of new, modified and
        obsolete entries from the most recent list. The directory with the
        largest numerical name is used.
        Returns None if something goes wrong.

        Contents of the data/status dir (20031013 would be used);:

            drwxrwxr-x   2 1002     sysadmin     512 Oct  6 18:28 20031006
            drwxrwxr-x   2 1002     sysadmin     512 Oct 14 02:14 20031013
            -rw-r--r--   1 1002     sysadmin    1327 Mar 12  2001 README

        """
    path = self.pdb_server + '/pub/pdb/data/status/latest/'
    added = self.get_status_list(path + 'added.pdb')
    modified = self.get_status_list(path + 'modified.pdb')
    obsolete = self.get_status_list(path + 'obsolete.pdb')
    return [added, modified, obsolete]