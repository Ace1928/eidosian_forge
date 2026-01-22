import operator
import warnings
def fractional_presence(label):
    return lambda x, y: abs(1.0 / len(x) - 1.0 / len(y)) * (label in x and label in y) or 0.0 * (label not in x and label not in y) or abs(1.0 / len(x)) * (label in x and label not in y) or 1.0 / len(y) * (label not in x and label in y)