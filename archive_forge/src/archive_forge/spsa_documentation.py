import numpy as np
from pennylane.measurements import Shots
Update the variables to take a single optimization step.

        Args:
            grad (tuple [array]): the gradient approximation of the objective
                function at point :math:`\hat{\theta}_{k}`
            args (tuple): the current value of the variables :math:`\hat{\theta}_{k}`

        Returns:
            list [array]: the new values :math:`\hat{\theta}_{k+1}`