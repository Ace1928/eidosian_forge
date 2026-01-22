from math import cos, exp, sin, sqrt, atan2
def dist_cie94(rgb1, rgb2):
    """
    Determine distance between two rgb colors using the CIE94 algorithm.

    :arg tuple rgb1: RGB color definition
    :arg tuple rgb2: RGB color definition
    :returns: Square of the distance between provided colors
    :rtype: float

    For efficiency, the square of the distance is returned
    which is sufficient for comparisons
    """
    l_1, a_1, b_1 = rgb_to_lab(*rgb1)
    l_2, a_2, b_2 = rgb_to_lab(*rgb2)
    s_l = k_l = k_c = k_h = 1
    k_1 = 0.045
    k_2 = 0.015
    delta_l = l_1 - l_2
    delta_a = a_1 - a_2
    delta_b = b_1 - b_2
    c_1 = sqrt(a_1 ** 2 + b_1 ** 2)
    c_2 = sqrt(a_2 ** 2 + b_2 ** 2)
    delta_c = c_1 - c_2
    delta_h = sqrt(delta_a ** 2 + delta_b ** 2 + delta_c ** 2)
    s_c = 1 + k_1 * c_1
    s_h = 1 + k_2 * c_1
    return (delta_l / (k_l * s_l)) ** 2 + (delta_c / (k_c * s_c)) ** 2 + (delta_h / (k_h * s_h)) ** 2