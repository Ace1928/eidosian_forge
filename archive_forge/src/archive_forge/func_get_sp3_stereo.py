import logging
import re
from rdkit import Chem
from rdkit.Chem import inchi
def get_sp3_stereo(self):
    """ retrieve sp3 stereo information
        return a 4-item tuple containing
        1) Number of stereocenters detected. If 0, the remaining items of the tuple = None
        2) Number of undefined stereocenters. Must be smaller or equal to above
        3) True if the molecule is a meso form (with chiral centers and a plane of symmetry)
        4) Comma-separated list of internal atom numbers with sp3 stereochemistry
        """
    sp3_stereo = {}
    for con_layer in self.parsed_inchi:
        for fixed_layer in self.parsed_inchi[con_layer]:
            sp3_stereo[fixed_layer] = {}
            for iso_layer in self.parsed_inchi[con_layer][fixed_layer]:
                sp3_stereo[fixed_layer][iso_layer] = {}
                stereo_match = stereo_re.match(self.parsed_inchi[con_layer][fixed_layer][iso_layer])
                stereo_all_match = stereo_all_re.match(self.parsed_inchi[con_layer][fixed_layer][iso_layer])
                num_stereo = 0
                num_undef_stereo = 0
                is_meso = False
                stereo = ''
                if stereo_match:
                    stereo = stereo_match.group(1)
                elif stereo_all_match:
                    stereo = stereo_all_match.group(1)
                    is_meso = len(defined_stereo_re.findall(stereo)) > 1
                num_stereo = len(all_stereo_re.findall(stereo))
                num_undef_stereo = len(undef_stereo_re.findall(stereo))
                inchi_layer = self.parsed_inchi[con_layer][fixed_layer][iso_layer]
                is_meso = is_meso or (num_undef_stereo > 1 and _is_achiral_by_symmetry(inchi_layer))
                sp3_stereo[fixed_layer][iso_layer] = (num_stereo, num_undef_stereo, is_meso, stereo)
    return sp3_stereo