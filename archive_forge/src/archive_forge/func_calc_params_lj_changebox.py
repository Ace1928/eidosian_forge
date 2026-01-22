import pytest
import numpy as np
from ase import Atoms
def calc_params_lj_changebox(spec, lj_cutoff, eps_orig, eps_modified):

    def lj_pair_style_coeff_lines(lj_cutoff, eps):
        return [f'pair_style lj/cut {lj_cutoff}', f'pair_coeff * * {eps} 1']
    calc_params = {}
    calc_params['lmpcmds'] = lj_pair_style_coeff_lines(lj_cutoff, eps_orig)
    calc_params['atom_types'] = {spec: 1}
    calc_params['log_file'] = 'test.log'
    calc_params['keep_alive'] = True
    calc_params['post_changebox_cmds'] = lj_pair_style_coeff_lines(lj_cutoff, eps_modified)
    return calc_params