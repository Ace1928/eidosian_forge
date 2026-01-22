from warnings import warn
import logging
from rdkit import Chem
Enumerate all possible resonance forms and return them as a list.

        :param mol: The input molecule.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: A list of all possible resonance forms of the molecule.
        :rtype: list of :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        