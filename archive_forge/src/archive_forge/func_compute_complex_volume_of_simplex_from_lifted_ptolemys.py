from ...sage_helper import _within_sage, sage_method
@sage_method
def compute_complex_volume_of_simplex_from_lifted_ptolemys(index, ptolemys):
    """
    Given lifted Ptolemy coordinates for a triangulation (as dictionary),
    compute the complex volume contribution by the simplex with given index.
    """
    c_1100 = ptolemys['c_1100_%d' % index]
    c_1010 = ptolemys['c_1010_%d' % index]
    c_1001 = ptolemys['c_1001_%d' % index]
    c_0110 = ptolemys['c_0110_%d' % index]
    c_0101 = ptolemys['c_0101_%d' % index]
    c_0011 = ptolemys['c_0011_%d' % index]
    w0 = c_1010 + c_0101 - c_1001 - c_0110
    w1 = c_1001 + c_0110 - c_1100 - c_0011
    return compute_Neumanns_Rogers_dilog_from_flattening_w0_w1(w0, w1)