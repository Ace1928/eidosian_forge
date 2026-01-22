import inspect
import numpy as np
from .bdf import BDF
from .radau import Radau
from .rk import RK23, RK45, DOP853
from .lsoda import LSODA
from scipy.optimize import OptimizeResult
from .common import EPS, OdeSolution
from .base import OdeSolver
def find_active_events(g, g_new, direction):
    """Find which event occurred during an integration step.

    Parameters
    ----------
    g, g_new : array_like, shape (n_events,)
        Values of event functions at a current and next points.
    direction : ndarray, shape (n_events,)
        Event "direction" according to the definition in `solve_ivp`.

    Returns
    -------
    active_events : ndarray
        Indices of events which occurred during the step.
    """
    g, g_new = (np.asarray(g), np.asarray(g_new))
    up = (g <= 0) & (g_new >= 0)
    down = (g >= 0) & (g_new <= 0)
    either = up | down
    mask = up & (direction > 0) | down & (direction < 0) | either & (direction == 0)
    return np.nonzero(mask)[0]