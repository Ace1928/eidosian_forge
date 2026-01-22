import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
import numpy as np
def chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    """Frequency-swept cosine generator.

    In the following, 'Hz' should be interpreted as 'cycles per unit';
    there is no requirement here that the unit is one second.  The
    important distinction is that the units of rotation are cycles, not
    radians. Likewise, `t` could be a measurement of space instead of time.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the waveform.
    f0 : float
        Frequency (e.g. Hz) at time t=0.
    t1 : float
        Time at which `f1` is specified.
    f1 : float
        Frequency (e.g. Hz) of the waveform at time `t1`.
    method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
        Kind of frequency sweep.  If not given, `linear` is assumed.  See
        Notes below for more details.
    phi : float, optional
        Phase offset, in degrees. Default is 0.
    vertex_zero : bool, optional
        This parameter is only used when `method` is 'quadratic'.
        It determines whether the vertex of the parabola that is the graph
        of the frequency is at t=0 or t=t1.

    Returns
    -------
    y : ndarray
        A numpy array containing the signal evaluated at `t` with the
        requested time-varying frequency.  More precisely, the function
        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.

    Examples
    --------
    The following will be used in the examples:

    >>> from cupyx.scipy.signal import chirp, spectrogram
    >>> import matplotlib.pyplot as plt
    >>> import cupy as cp

    For the first example, we'll plot the waveform for a linear chirp
    from 6 Hz to 1 Hz over 10 seconds:

    >>> t = cupy.linspace(0, 10, 5001)
    >>> w = chirp(t, f0=6, f1=1, t1=10, method='linear')
    >>> plt.plot(cupy.asnumpy(t), cupy.asnumpy(w))
    >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")
    >>> plt.xlabel('t (sec)')
    >>> plt.show()

    For the remaining examples, we'll use higher frequency ranges,
    and demonstrate the result using `cupyx.scipy.signal.spectrogram`.
    We'll use a 10 second interval sampled at 8000 Hz.

    >>> fs = 8000
    >>> T = 10
    >>> t = cupy.linspace(0, T, T*fs, endpoint=False)

    Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds
    (vertex of the parabolic curve of the frequency is at t=0):

    >>> w = chirp(t, f0=1500, f1=250, t1=10, method='quadratic')
    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
    ...                           nfft=2048)
    >>> plt.pcolormesh(cupy.asnumpy(tt), cupy.asnumpy(ff[:513]),
                       cupy.asnumpy(Sxx[:513]), cmap='gray_r')
    >>> plt.title('Quadratic Chirp, f(0)=1500, f(10)=250')
    >>> plt.xlabel('t (sec)')
    >>> plt.ylabel('Frequency (Hz)')
    >>> plt.grid()
    >>> plt.show()
    """
    t = cupy.asarray(t)
    if cupy.issubdtype(t.dtype, cupy.int_):
        t = t.astype(cupy.float64)
    phi *= np.pi / 180
    type = 'real'
    if method in ['linear', 'lin', 'li']:
        if type == 'real':
            return _chirp_phase_lin_kernel_real(t, f0, t1, f1, phi)
        elif type == 'complex':
            phase = cupy.empty(t.shape, dtype=cupy.complex64)
            if np.issubclass_(t.dtype, np.float64):
                phase = cupy.empty(t.shape, dtype=cupy.complex128)
            _chirp_phase_lin_kernel_cplx(t, f0, t1, f1, phi, phase)
            return phase
        else:
            raise NotImplementedError('No kernel for type {}'.format(type))
    elif method in ['quadratic', 'quad', 'q']:
        return _chirp_phase_quad_kernel(t, f0, t1, f1, phi, vertex_zero)
    elif method in ['logarithmic', 'log', 'lo']:
        if f0 * f1 <= 0.0:
            raise ValueError('For a logarithmic chirp, f0 and f1 must be nonzero and have the same sign.')
        return _chirp_phase_log_kernel(t, f0, t1, f1, phi)
    elif method in ['hyperbolic', 'hyp']:
        if f0 == 0 or f1 == 0:
            raise ValueError('For a hyperbolic chirp, f0 and f1 must be nonzero.')
        return _chirp_phase_hyp_kernel(t, f0, t1, f1, phi)
    else:
        raise ValueError("method must be 'linear', 'quadratic', 'logarithmic', or 'hyperbolic', but a value of %r was given." % method)