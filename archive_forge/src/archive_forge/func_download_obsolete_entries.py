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
def download_obsolete_entries(self, listfile=None, file_format=None):
    """Retrieve all obsolete PDB entries not present in local obsolete PDB copy.

        :param listfile: filename to which all PDB codes will be written (optional)

        :param file_format: file format. Available options:
            "mmCif" (default, PDBx/mmCif file),
            "pdb" (format PDB),
            "xml" (PMDML/XML format),

        NOTE. The default download format has changed from PDB to PDBx/mmCif
        """
    file_format = self._print_default_format_warning(file_format)
    entries = self.get_all_obsolete()
    for pdb_code in entries:
        self.retrieve_pdb_file(pdb_code, obsolete=True, file_format=file_format)
    if listfile:
        with open(listfile, 'w') as outfile:
            outfile.writelines((x + '\n' for x in entries))