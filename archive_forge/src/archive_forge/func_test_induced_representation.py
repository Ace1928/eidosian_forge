from snappy import Manifold, pari, ptolemy
from snappy.ptolemy import solutions_from_magma, Flattenings, parse_solutions
from snappy.ptolemy.processFileBase import get_manifold
from snappy.ptolemy import __path__ as ptolemy_paths
from snappy.ptolemy.coordinates import PtolemyCannotBeCheckedError
from snappy.sage_helper import _within_sage, doctest_modules
from snappy.pari import pari
import bz2
import os
import sys
def test_induced_representation():
    M = ManifoldGetter('m015')
    variety__sl2_c1 = M.ptolemy_variety(2, obstruction_class=1)
    variety__sl3_c0 = M.ptolemy_variety(3, obstruction_class=0)
    solutions__sl2_c1 = compute_using_precomputed_magma(variety__sl2_c1, dir=testing_files_generalized_directory)
    solutions__sl3_c0 = compute_using_precomputed_magma(variety__sl3_c0, dir=testing_files_generalized_directory)
    got_exception = False
    try:
        solutions__sl3_c0[0].cross_ratios().is_real(epsilon=1e-80)
    except:
        got_exception = True
    assert got_exception, 'Expected error when calling is_real on exact solution'
    numbers_all_and_real = []
    for component in solutions__sl3_c0:
        number_real = 0
        number_all = 0
        for z in component.cross_ratios_numerical():
            if z.is_real(epsilon=1e-80):
                number_real += 1
            number_all += 1
        numbers_all_and_real.append((number_all, number_real))
    numbers_all_and_real.sort()
    expected_numbers = [(3, 1), (4, 2), (6, 0)]
    assert numbers_all_and_real == expected_numbers, 'Order of components and number of real solutions is off'
    number_psl2 = 0
    for component in solutions__sl3_c0:
        if component.cross_ratios().is_induced_from_psl2():
            number_psl2 += 1
    assert number_psl2 == 1, 'Only one component can come from psl2'
    number_psl2 = 0
    for component in solutions__sl3_c0:
        is_induced_from_psl2 = [z.is_induced_from_psl2(epsilon=1e-80) for z in component.cross_ratios_numerical()]
        if any(is_induced_from_psl2):
            number_psl2 += 1
            assert all(is_induced_from_psl2), 'Mixed up is_induced_from_psl2'
    assert number_psl2 == 1, 'Only one component can come from psl2 (numerically)'
    got_exception = False
    try:
        solutions__sl3_c0[0].cross_ratios().induced_representation(3)
    except:
        got_exception = True
    assert got_exception, 'Expected error when calling induced_representation on sl3'
    m015_volume = pari('2.828122088330783162763898809276634942770981317300649477043520327258802548322471630936947017929999108')
    z = solutions__sl2_c1[0].cross_ratios().induced_representation(3)
    assert z.is_induced_from_psl2(), 'induced_representation failed to be detected as being induced'
    assert z.check_against_manifold, 'induced_representation fails to be valid'
    for v in z.volume_numerical():
        assert v.abs() < 1e-80 or (v.abs() - 4 * m015_volume).abs() < 1e-80, 'Did not get expected voluem for induced representation'
    for z in solutions__sl2_c1[0].cross_ratios_numerical():
        v = z.induced_representation(3).volume_numerical()
        assert v.abs() < 1e-80 or (v.abs() - 4 * m015_volume).abs() < 1e-80, 'Did not get expected voluem for induced representation'