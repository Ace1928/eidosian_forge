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
def get_all_assemblies(self, file_format='mmCif'):
    """Retrieve the list of PDB entries with an associated bio assembly.

        The requested list will be cached to avoid multiple calls to the FTP
        server.

        :type  file_format: str, optional
        :param file_format: format in which to download the entries. Available
            options are "mmCif" or "pdb". Defaults to mmCif.
        """
    if hasattr(self, 'assemblies') and self.assemblies:
        if self._verbose:
            print('Retrieving cached list of assemblies.')
        return self.assemblies
    if self._verbose:
        print('Retrieving list of assemblies. This might take a while.')
    idx = self.pdb_server.find('://')
    if idx >= 0:
        ftp = ftplib.FTP(self.pdb_server[idx + 3:])
    else:
        ftp = ftplib.FTP(self.pdb_server)
    ftp.login()
    if file_format.lower() == 'mmcif':
        ftp.cwd('/pub/pdb/data/assemblies/mmCIF/all/')
        re_name = re.compile('(\\d[0-9a-z]{3})-assembly(\\d+).cif.gz')
    elif file_format.lower() == 'pdb':
        ftp.cwd('/pub/pdb/data/biounit/PDB/all/')
        re_name = re.compile('(\\d[0-9a-z]{3}).pdb(\\d+).gz')
    else:
        msg = "file_format for assemblies must be 'pdb' or 'mmCif'"
        raise ValueError(msg)
    response = []
    ftp.retrlines('NLST', callback=response.append)
    all_assemblies = []
    for line in response:
        if line.endswith('.gz'):
            match = re_name.findall(line)
            try:
                if len(match):
                    entry, assembly = match[0]
            except ValueError:
                pass
            else:
                all_assemblies.append((entry, assembly))
    self.assemblies = all_assemblies
    return all_assemblies