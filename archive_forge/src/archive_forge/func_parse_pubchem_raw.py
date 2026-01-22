from collections import namedtuple
import warnings
import urllib.request
from urllib.error import URLError, HTTPError
import json
from io import StringIO, BytesIO
from ase.io import read
def parse_pubchem_raw(data):
    """
    a helper function for parsing the returned pubchem entries

    Parameters:
        data (str):
            the raw output from pubchem in string form

    returns:
        atoms (ASE Atoms Object):
            An ASE atoms obejct containing the information from
            pubchem
        pubchem_data (dict):
            a dictionary containing the non-structural information
            from pubchem

    """
    if 'PUBCHEM_COMPOUND_CID' not in data:
        raise Exception('There was a problem with the data returned by PubChem')
    f_like = StringIO(data)
    atoms = read(f_like, format='sdf')
    pubchem_data = {}
    other_info = data.split('END\n')[1]
    other_info = other_info.split('$')[0]
    other_info = other_info.split('> <')
    for data_field in other_info:
        if data_field == '':
            continue
        field_name, entry_value = data_field.split('>\n')
        entry_value = entry_value.splitlines()
        entry_value = [a for a in entry_value if a != '']
        if len(entry_value) == 1:
            entry_value = entry_value[0]
        pubchem_data[field_name] = entry_value
    if 'PUBCHEM_MMFF94_PARTIAL_CHARGES' in pubchem_data.keys():
        charges = pubchem_data['PUBCHEM_MMFF94_PARTIAL_CHARGES'][1:]
        atom_charges = [0.0] * len(atoms)
        for charge in charges:
            i, charge = charge.split()
            atom_charges[int(i) - 1] = float(charge)
        atoms.set_initial_charges(atom_charges)
    return (atoms, pubchem_data)