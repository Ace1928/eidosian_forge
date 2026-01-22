import logging
import math
import numpy as np
from ase.utils import longsum
def handle_args(self, x_start, dirn, a_max, a_min, a1, func_start, func_old, func_prime_start, maxstep):
    """Verify passed parameters and set appropriate attributes accordingly.

        A suitable value for the initial step-length guess will be either
        verified or calculated, stored in the attribute self.a_start, and
        returned.

        Args:
            The args should be identical to those of self.run().

        Returns:
            The suitable initial step-length guess a_start

        Raises:
            ValueError for problems with arguments

        """
    self.a_max = a_max
    self.a_min = a_min
    self.x_start = x_start
    self.dirn = dirn
    self.func_old = func_old
    self.func_start = func_start
    self.func_prime_start = func_prime_start
    if a_max is None:
        a_max = 2.0
    if a_max < self.tol:
        logger.warning('a_max too small relative to tol. Reverting to default value a_max = 2.0 (twice the <ideal> step).')
        a_max = 2.0
    if self.a_min is None:
        self.a_min = 1e-10
    if func_start is None:
        logger.debug('Setting func_start')
        self.func_start = self.func(x_start)
    self.phi_prime_start = longsum(self.func_prime_start * self.dirn)
    if self.phi_prime_start >= 0:
        logger.error('Passed direction which is not downhill. Aborting...')
        raise ValueError('Direction is not downhill.')
    elif math.isinf(self.phi_prime_start):
        logger.error('Passed func_prime_start and dirn which are too big. Aborting...')
        raise ValueError('func_prime_start and dirn are too big.')
    if a1 is None:
        if func_old is not None:
            a1 = 2 * (self.func_start - self.func_old) / self.phi_prime_start
            logger.debug('Interpolated quadratic, obtained a1 = %e', a1)
    if a1 is None or a1 > a_max:
        logger.debug('a1 greater than a_max. Reverting to default value a1 = 1.0')
        a1 = 1.0
    if a1 is None or a1 < self.tol:
        logger.debug('a1 is None or a1 < self.tol. Reverting to default value a1 = 1.0')
        a1 = 1.0
    if a1 is None or a1 < self.a_min:
        logger.debug('a1 is None or a1 < a_min. Reverting to default value a1 = 1.0')
        a1 = 1.0
    if maxstep is None:
        maxstep = 0.2
    logger.debug('maxstep = %e', maxstep)
    r = np.reshape(dirn, (-1, 3))
    steplengths = ((a1 * r) ** 2).sum(1) ** 0.5
    maxsteplength = np.max(steplengths)
    if maxsteplength >= maxstep:
        a1 *= maxstep / maxsteplength
        logger.debug('Rescaled a1 to fulfill maxstep criterion')
    self.a_start = a1
    logger.debug('phi_start = %e, phi_prime_start = %e', self.func_start, self.phi_prime_start)
    logger.debug('func_start = %s, self.func_old = %s', self.func_start, self.func_old)
    logger.debug('a1 = %e, a_max = %e, a_min = %e', a1, a_max, self.a_min)
    return a1