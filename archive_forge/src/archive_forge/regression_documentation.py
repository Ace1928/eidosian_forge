import numbers
import numpy as np
from ...utils import _safe_indexing, check_matplotlib_support, check_random_state
Plot the prediction error given the true and predicted targets.

        For general information regarding `scikit-learn` visualization tools,
        read more in the :ref:`Visualization Guide <visualizations>`.
        For details regarding interpreting these plots, refer to the
        :ref:`Model Evaluation Guide <visualization_regression_evaluation>`.

        .. versionadded:: 1.2

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True target values.

        y_pred : array-like of shape (n_samples,)
            Predicted target values.

        kind : {"actual_vs_predicted", "residual_vs_predicted"},                 default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1000 samples or less will be displayed.

        random_state : int or RandomState, default=None
            Controls the randomness when `subsample` is not `None`.
            See :term:`Glossary <random_state>` for details.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        scatter_kwargs : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.scatter`
            call.

        line_kwargs : dict, default=None
            Dictionary with keyword passed to the `matplotlib.pyplot.plot`
            call to draw the optimal line.

        Returns
        -------
        display : :class:`~sklearn.metrics.PredictionErrorDisplay`
            Object that stores the computed values.

        See Also
        --------
        PredictionErrorDisplay : Prediction error visualization for regression.
        PredictionErrorDisplay.from_estimator : Prediction error visualization
            given an estimator and some data.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import PredictionErrorDisplay
        >>> X, y = load_diabetes(return_X_y=True)
        >>> ridge = Ridge().fit(X, y)
        >>> y_pred = ridge.predict(X)
        >>> disp = PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred)
        >>> plt.show()
        