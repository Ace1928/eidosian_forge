from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase
from . import utilities
import snappy
import re
import sys
import tempfile
import subprocess
import shutil
def decomposition_from_magma(text):
    py_eval = processFileBase.get_py_eval(text)
    manifold_thunk = processFileBase.get_manifold_thunk(text)
    untyped_decomposition = processFileBase.find_section(text, 'IDEAL=DECOMPOSITION')
    primary_decomposition = processFileBase.find_section(text, 'PRIMARY=DECOMPOSITION')
    radical_decomposition = processFileBase.find_section(text, 'RADICAL=DECOMPOSITION')
    if untyped_decomposition:
        decomposition = untyped_decomposition[0]
    elif primary_decomposition:
        decomposition = primary_decomposition[0]
    elif radical_decomposition:
        decomposition = radical_decomposition[0]
    else:
        raise ValueError('File not recognized as magma output (missing primary decomposition or radical decomposition)')
    decomposition = utilities.join_long_lines_deleting_whitespace(decomposition)
    decomposition = processFileBase.remove_outer_square_brackets(decomposition)
    decomposition_comps = [c.strip() for c in decomposition.split(']')]
    decomposition_components = [c + ']' for c in decomposition_comps if c]
    free_variables_section = processFileBase.find_section(text, 'FREE=VARIABLES=IN=COMPONENTS')
    if free_variables_section:
        free_variables = eval(free_variables_section[0])
    else:
        free_variables = len(decomposition_components) * [None]
    witnesses_section = processFileBase.find_section(text, 'WITNESSES=FOR=COMPONENTS')
    if witnesses_section:
        witnesses_sections = processFileBase.find_section(witnesses_section[0], 'WITNESSES')
    else:
        witnesses_sections = len(decomposition_components) * ['']
    genuses_section = processFileBase.find_section(text, 'GENUSES=FOR=COMPONENTS')
    if genuses_section:
        genuses_sections = processFileBase.find_section(genuses_section[0], 'GENUS=FOR=COMPONENT')
    else:
        genuses_sections = len(decomposition_components) * ['']

    def process_match(i, comp, free_vars, witnesses_txt, genus_txt):
        if i != 0:
            if not comp[0] == ',':
                raise ValueError('Parsing decomposition, expected separating comma.')
            comp = comp[1:].strip()
        if genus_txt.strip():
            genus = int(genus_txt)
        else:
            genus = None
        witnesses_txts = processFileBase.find_section(witnesses_txt, 'WITNESS')
        witnesses = [_parse_ideal_groebner_basis(utilities.join_long_lines_deleting_whitespace(t).strip(), py_eval, manifold_thunk, free_vars, [], genus) for t in witnesses_txts]
        return _parse_ideal_groebner_basis(comp, py_eval, manifold_thunk, free_vars, witnesses, genus)
    return utilities.MethodMappingList([process_match(i, comp, free_vars, witnesses, genus_txt) for i, (comp, free_vars, witnesses, genus_txt) in enumerate(zip(decomposition_components, free_variables, witnesses_sections, genuses_sections))])