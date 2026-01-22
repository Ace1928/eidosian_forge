import math
def norm_logsf(x):
    try:
        return math.log(1 - norm_cdf(x))
    except ValueError:
        return float('-inf')