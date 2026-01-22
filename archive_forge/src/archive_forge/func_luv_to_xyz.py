import operator
import math
def luv_to_xyz(triple):
    L, U, V = triple
    if L == 0:
        return [0.0, 0.0, 0.0]
    varY = f_inv((L + 16.0) / 116.0)
    varU = U / (13.0 * L) + refU
    varV = V / (13.0 * L) + refV
    Y = varY * refY
    X = 0.0 - 9.0 * Y * varU / ((varU - 4.0) * varV - varU * varV)
    Z = (9.0 * Y - 15.0 * varV * Y - varV * X) / (3.0 * varV)
    return [X, Y, Z]