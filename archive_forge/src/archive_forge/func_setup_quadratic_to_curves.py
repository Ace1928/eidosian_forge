from .qu2cu import *
from fontTools.cu2qu import curve_to_quadratic
import random
import timeit
def setup_quadratic_to_curves():
    curves = generate_curves(NUM_CURVES)
    quadratics = [curve_to_quadratic(curve, MAX_ERR) for curve in curves]
    return (quadratics, MAX_ERR)