import numpy as np
Transform parameters of the standardized model to the original model

        Parameters
        ----------
        params : ndarray
            parameters estimated with the standardized model

        Returns
        -------
        params_new : ndarray
            parameters transformed to the parameterization of the original
            model
        