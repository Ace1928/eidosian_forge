import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import PPBuilder
def get_resname_list(self):
    """Get residue list.

        :return: the residue names
        :rtype: [string, string,...]
        """
    return self.resname_list