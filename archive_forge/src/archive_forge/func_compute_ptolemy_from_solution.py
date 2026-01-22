from snappy.dev.extended_ptolemy import extended
from snappy.dev.extended_ptolemy import giac_rur
from snappy.dev.extended_ptolemy.complexVolumesClosed import evaluate_at_roots
from snappy.ptolemy.coordinates import PtolemyCoordinates, CrossRatios
def compute_ptolemy_from_solution(I, solution, dict_value):
    sign, m_count, l_count, name = dict_value
    R = I.ring()
    m = solution[R('M')] if m_count >= 0 else solution[R('m')]
    l = solution[R('L')] if l_count >= 0 else solution[R('l')]
    return sign * m ** abs(m_count) * l ** abs(l_count) * solution[R(name)]