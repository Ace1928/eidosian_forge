from math import cos, exp, sin, sqrt, atan2
def xyz_to_lab(x_val, y_val, z_val):
    """
    Convert XYZ color to CIE-Lab color.

    :arg float x_val: XYZ value of X.
    :arg float y_val: XYZ value of Y.
    :arg float z_val: XYZ value of Z.
    :returns: Tuple (L, a, b) representing CIE-Lab color
    :rtype: tuple

    D65/2° standard illuminant
    """
    xyz = []
    for val, ref in ((x_val, 95.047), (y_val, 100.0), (z_val, 108.883)):
        val /= ref
        val = pow(val, 1 / 3.0) if val > 0.008856 else 7.787 * val + 16 / 116.0
        xyz.append(val)
    x_val, y_val, z_val = xyz
    cie_l = 116 * y_val - 16
    cie_a = 500 * (x_val - y_val)
    cie_b = 200 * (y_val - z_val)
    return (cie_l, cie_a, cie_b)