import io
import os
import pathlib
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
@classmethod
def from_rdkit(cls, data_or_path: 'RDKitDataType', caption: Optional[str]=None, convert_to_3d_and_optimize: bool=True, mmff_optimize_molecule_max_iterations: int=200) -> 'Molecule':
    """Convert RDKit-supported file/object types to wandb.Molecule.

        Arguments:
            data_or_path: (string, rdkit.Chem.rdchem.Mol)
                Molecule can be initialized from a file name or an rdkit.Chem.rdchem.Mol object.
            caption: (string)
                Caption associated with the molecule for display.
            convert_to_3d_and_optimize: (bool)
                Convert to rdkit.Chem.rdchem.Mol with 3D coordinates.
                This is an expensive operation that may take a long time for complicated molecules.
            mmff_optimize_molecule_max_iterations: (int)
                Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule
        """
    rdkit_chem = util.get_module('rdkit.Chem', required='wandb.Molecule needs the rdkit-pypi package. To get it, run "pip install rdkit-pypi".')
    rdkit_chem_all_chem = util.get_module('rdkit.Chem.AllChem', required='wandb.Molecule needs the rdkit-pypi package. To get it, run "pip install rdkit-pypi".')
    if isinstance(data_or_path, str):
        path = pathlib.Path(data_or_path)
        extension = path.suffix.split('.')[-1]
        if extension not in Molecule.SUPPORTED_RDKIT_TYPES:
            raise ValueError('Molecule.from_rdkit only supports files of the type: ' + ', '.join(Molecule.SUPPORTED_RDKIT_TYPES))
        if extension == 'sdf':
            with rdkit_chem.SDMolSupplier(data_or_path) as supplier:
                molecule = next(supplier)
        else:
            molecule = getattr(rdkit_chem, f'MolFrom{extension.capitalize()}File')(data_or_path)
    elif isinstance(data_or_path, rdkit_chem.rdchem.Mol):
        molecule = data_or_path
    else:
        raise ValueError('Data must be file name or an rdkit.Chem.rdchem.Mol object')
    if convert_to_3d_and_optimize:
        molecule = rdkit_chem.AddHs(molecule)
        rdkit_chem_all_chem.EmbedMolecule(molecule)
        rdkit_chem_all_chem.MMFFOptimizeMolecule(molecule, maxIters=mmff_optimize_molecule_max_iterations)
    pdb_block = rdkit_chem.rdmolfiles.MolToPDBBlock(molecule)
    return cls(io.StringIO(pdb_block), caption=caption, file_type='pdb')