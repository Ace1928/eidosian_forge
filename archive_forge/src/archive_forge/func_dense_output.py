import numpy as np
def dense_output(self):
    """Compute a local interpolant over the last successful step.

        Returns
        -------
        sol : `DenseOutput`
            Local interpolant over the last successful step.
        """
    if self.t_old is None:
        raise RuntimeError('Dense output is available after a successful step was made.')
    if self.n == 0 or self.t == self.t_old:
        return ConstantDenseOutput(self.t_old, self.t, self.y)
    else:
        return self._dense_output_impl()