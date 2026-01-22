import warnings
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
def init_chain(self, chain_id):
    """Create a new Chain object with given id.

        Arguments:
         - chain_id - string

        """
    if self.model.has_id(chain_id):
        self.chain = self.model[chain_id]
        warnings.warn('WARNING: Chain %s is discontinuous at line %i.' % (chain_id, self.line_counter), PDBConstructionWarning)
    else:
        self.chain = Chain(chain_id)
        self.model.add(self.chain)