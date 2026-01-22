import numpy as np
def fermidistribution(energy, kt):
    assert kt >= 0.0, 'Negative temperature encountered!'
    if kt == 0:
        if isinstance(energy, float):
            return int(energy / 2.0 <= 0)
        else:
            return (energy / 2.0 <= 0).astype(int)
    else:
        return 1.0 / (1.0 + np.exp(energy / kt))