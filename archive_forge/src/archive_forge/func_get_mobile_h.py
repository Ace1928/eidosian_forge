import logging
import re
from rdkit import Chem
from rdkit.Chem import inchi
def get_mobile_h(self):
    """ retrieve mobile H (tautomer) information
        return a 2-item tuple containing
        1) Number of mobile hydrogen groups detected. If 0, next item = '' 
        2) List of groups   
        """
    mobile_h = {}
    for con_layer in self.parsed_inchi:
        for fixed_layer in self.parsed_inchi[con_layer]:
            mobile_h[fixed_layer] = {}
            for iso_layer in self.parsed_inchi[con_layer][fixed_layer]:
                num_groups = 0
                mobile_h_groups = ''
                h_layer_match = h_layer_re.match(self.parsed_inchi[con_layer][fixed_layer][iso_layer])
                if h_layer_match:
                    mobile_h_matches = mobile_h_group_re.findall(h_layer_match.group(1))
                    num_groups = len(mobile_h_matches)
                    mobile_h_groups = ','.join(mobile_h_matches)
                mobile_h[fixed_layer][iso_layer] = (num_groups, mobile_h_groups)
    return mobile_h