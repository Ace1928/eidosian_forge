def butter_lp(n, Wn):
    """
    Lowpass Butterworth digital filter design.

    This computes the same result as scipy.signal.butter(n, Wn, output='zpk'),
    but it uses mpmath, and the results are returned in lists instead of NumPy
    arrays.
    """
    zeros = []
    poles = _butter_analog_poles(n)
    k = 1
    fs = 2
    warped = 2 * fs * mpmath.tan(mpmath.pi * Wn / fs)
    z, p, k = _zpklp2lp(zeros, poles, k, wo=warped)
    z, p, k = _zpkbilinear(z, p, k, fs=fs)
    return (z, p, k)