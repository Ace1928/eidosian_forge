import os
from copy import deepcopy
from ase.io import read
from ase.calculators.calculator import ReadError
from ase.calculators.calculator import FileIOCalculator
def update_parameter(oldpar, newpar):
    """Update each section of parameter (oldpar) using newpar keys and values.
    If section of newpar exist in oldpar,
        - Replace the section_name with newpar's section_name if oldvar section_name type is not dict.
        - Append the section_name with newpar's section_name if oldvar section_name type is list.
        - If oldpar section_name type is dict, it is subsection. So call update_parameter again.
    otherwise, add the parameter section and section_name from newpar.

    Parameters
    ==========
    oldpar: dictionary of original parameters to be updated.
    newpar: dictionary containing parameter section and values to update.

    Return
    ======
    Updated parameter dictionary.
    """
    for section, section_param in newpar.items():
        if section in oldpar:
            if isinstance(section_param, dict):
                oldpar[section] = update_parameter(oldpar[section], section_param)
            else:
                oldpar[section] = section_param
        else:
            oldpar[section] = section_param
    return oldpar