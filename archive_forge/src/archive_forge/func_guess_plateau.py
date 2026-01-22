import numpy as np
def guess_plateau(x, y):
    """Given two axes returns a guess of the plateau point.

    The plateau point is defined as the x point where the y point
    is near one standard deviation of the differences between the y points to
    the maximum y value. If such point is not found or x and y have
    different lengths the function returns zero.
    """
    if len(x) != len(y):
        return 0
    diffs = []
    indexes = range(len(y))
    for i in indexes:
        if i + 1 not in indexes:
            continue
        diffs.append(y[i + 1] - y[i])
    diffs = np.array(diffs)
    ymax = y[-1]
    for i in indexes:
        if y[i] > ymax - diffs.std() and y[i] < ymax + diffs.std():
            ymax = y[i]
            break
    return ymax