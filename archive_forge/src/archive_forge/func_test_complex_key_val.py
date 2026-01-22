from pathlib import Path
import numpy as np
import pytest
import ase.io
from ase.io import extxyz
from ase.atoms import Atoms
from ase.build import bulk
from ase.io.extxyz import escape
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixCartesian
from ase.stress import full_3x3_to_voigt_6_stress
from ase.build import molecule
def test_complex_key_val():
    complex_xyz_string = ' str=astring quot="quoted value" quote_special="a_to_Z_$%%^&*" escaped_quote="esc\\"aped" true_value false_value = F integer=22 floating=1.1 int_array={1 2 3} float_array="3.3 4.4" virial="1 4 7 2 5 8 3 6 9" not_a_3x3_array="1 4 7 2 5 8 3 6 9" Lattice="  4.3  0.0 0.0 0.0  3.3 0.0 0.0 0.0  7.0 " scientific_float=1.2e7 scientific_float_2=5e-6 scientific_float_array="1.2 2.2e3 4e1 3.3e-1 2e-2" not_array="1.2 3.4 text" bool_array={T F T F} bool_array_2=" T, F, T " not_bool_array=[T F S] unquoted_special_value=a_to_Z_$%%^&* 2body=33.3 hyphen-ated many_other_quotes="4 8 12" comma_separated="7, 4, -1" bool_array_commas=[T, T, F, T] Properties=species:S:1:pos:R:3 multiple_separators       double_equals=abc=xyz trailing "with space"="a value" space\\"="a value" f_str_looks_like_array="[[1, 2, 3], [4, 5, 6]]" f_float_array="_JSON [[1.5, 2, 3], [4, 5, 6]]" f_int_array="_JSON [[1, 2], [3, 4]]" f_bool_bare f_bool_value=F f_dict={_JSON {"a" : 1}} '
    expected_dict = {'str': 'astring', 'quot': 'quoted value', 'quote_special': u'a_to_Z_$%%^&*', 'escaped_quote': 'esc"aped', 'true_value': True, 'false_value': False, 'integer': 22, 'floating': 1.1, 'int_array': np.array([1, 2, 3]), 'float_array': np.array([3.3, 4.4]), 'virial': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'not_a_3x3_array': np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]), 'Lattice': np.array([[4.3, 0.0, 0.0], [0.0, 3.3, 0.0], [0.0, 0.0, 7.0]]), 'scientific_float': 12000000.0, 'scientific_float_2': 5e-06, 'scientific_float_array': np.array([1.2, 2200, 40, 0.33, 0.02]), 'not_array': '1.2 3.4 text', 'bool_array': np.array([True, False, True, False]), 'bool_array_2': np.array([True, False, True]), 'not_bool_array': 'T F S', 'unquoted_special_value': 'a_to_Z_$%%^&*', '2body': 33.3, 'hyphen-ated': True, 'many_other_quotes': np.array([4, 8, 12]), 'comma_separated': np.array([7, 4, -1]), 'bool_array_commas': np.array([True, True, False, True]), 'Properties': 'species:S:1:pos:R:3', 'multiple_separators': True, 'double_equals': 'abc=xyz', 'trailing': True, 'with space': 'a value', 'space"': 'a value', 'f_str_looks_like_array': '[[1, 2, 3], [4, 5, 6]]', 'f_float_array': np.array([[1.5, 2, 3], [4, 5, 6]]), 'f_int_array': np.array([[1, 2], [3, 4]]), 'f_bool_bare': True, 'f_bool_value': False, 'f_dict': {'a': 1}}
    parsed_dict = extxyz.key_val_str_to_dict(complex_xyz_string)
    np.testing.assert_equal(parsed_dict, expected_dict)
    key_val_str = extxyz.key_val_dict_to_str(expected_dict)
    parsed_dict = extxyz.key_val_str_to_dict(key_val_str)
    np.testing.assert_equal(parsed_dict, expected_dict)
    with open('complex.xyz', 'w', encoding='utf-8') as f_out:
        f_out.write('1\n{}\nH 1.0 1.0 1.0'.format(complex_xyz_string))
    complex_atoms = ase.io.read('complex.xyz')
    for key, value in expected_dict.items():
        if key in ['Properties', 'Lattice']:
            continue
        else:
            np.testing.assert_equal(complex_atoms.info[key], value)