from __future__ import annotations
import os
import sqlite3
import textwrap
from array import array
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from monty.design_patterns import cached_class
from pymatgen.core.operations import MagSymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.groups import SymmetryGroup, in_array_list
from pymatgen.symmetry.settings import JonesFaithfulTransformation
from pymatgen.util.string import transformation_to_string
def data_str(self, include_og=True):
    """Get description of all data, including information for OG setting.

        Returns:
            str.
        """
    desc = {}
    description = ''
    if self.jf != JonesFaithfulTransformation.from_transformation_str('a,b,c;0,0,0'):
        description += 'Non-standard setting: .....\n'
        description += repr(self.jf)
        description += '\n\nStandard setting information: \n'
    desc['magtype'] = self._data['magtype']
    desc['bns_number'] = '.'.join(map(str, self._data['bns_number']))
    desc['bns_label'] = self._data['bns_label']
    desc['og_id'] = f'\t\tOG: {'.'.join(map(str, self._data['og_number']))} {self._data['og_label']}' if include_og else ''
    desc['bns_operators'] = ' '.join((op_data['str'] for op_data in self._data['bns_operators']))
    desc['bns_lattice'] = ' '.join((lattice_data['str'] for lattice_data in self._data['bns_lattice'][3:])) if len(self._data['bns_lattice']) > 3 else ''
    desc['bns_wyckoff'] = '\n'.join([textwrap.fill(wyckoff_data['str'], initial_indent=wyckoff_data['label'] + '  ', subsequent_indent=' ' * len(wyckoff_data['label'] + '  '), break_long_words=False, break_on_hyphens=False) for wyckoff_data in self._data['bns_wyckoff']])
    desc['og_bns_transformation'] = f'OG-BNS Transform: ({self._data['og_bns_transform']})\n' if desc['magtype'] == 4 and include_og else ''
    bns_operators_prefix = f'Operators{(' (BNS)' if desc['magtype'] == 4 and include_og else '')}: '
    bns_wyckoff_prefix = f'Wyckoff Positions{(' (BNS)' if desc['magtype'] == 4 and include_og else '')}: '
    desc['bns_operators'] = textwrap.fill(desc['bns_operators'], initial_indent=bns_operators_prefix, subsequent_indent=' ' * len(bns_operators_prefix), break_long_words=False, break_on_hyphens=False)
    description += f'BNS: {desc['bns_number']} {desc['bns_label']}{desc['og_id']}\n{desc['og_bns_transformation']}{desc['bns_operators']}\n{bns_wyckoff_prefix}{desc['bns_lattice']}\n{desc['bns_wyckoff']}'
    if desc['magtype'] == 4 and include_og:
        desc['og_operators'] = ' '.join((op_data['str'] for op_data in self._data['og_operators']))
        desc['og_lattice'] = ' '.join((lattice_data['str'] for lattice_data in self._data['og_lattice']))
        desc['og_wyckoff'] = '\n'.join([textwrap.fill(wyckoff_data['str'], initial_indent=wyckoff_data['label'] + '  ', subsequent_indent=' ' * len(wyckoff_data['label'] + '  '), break_long_words=False, break_on_hyphens=False) for wyckoff_data in self._data['og_wyckoff']])
        og_operators_prefix = 'Operators (OG): '
        desc['og_operators'] = textwrap.fill(desc['og_operators'], initial_indent=og_operators_prefix, subsequent_indent=' ' * len(og_operators_prefix), break_long_words=False, break_on_hyphens=False)
        description += f'\n{desc['og_operators']}\nWyckoff Positions (OG): {desc['og_lattice']}\n{desc['og_wyckoff']}'
    elif desc['magtype'] == 4:
        description += '\nAlternative OG setting exists for this space group.'
    return description