import numpy as np
from patsy.util import have_pandas, no_pickling, assert_no_pickling
from patsy.state import stateful_transform
bs(x, df=None, knots=None, degree=3, include_intercept=False, lower_bound=None, upper_bound=None)

    Generates a B-spline basis for ``x``, allowing non-linear fits. The usual
    usage is something like::

      y ~ 1 + bs(x, 4)

    to fit ``y`` as a smooth function of ``x``, with 4 degrees of freedom
    given to the smooth.

    :arg df: The number of degrees of freedom to use for this spline. The
      return value will have this many columns. You must specify at least one
      of ``df`` and ``knots``.
    :arg knots: The interior knots to use for the spline. If unspecified, then
      equally spaced quantiles of the input data are used. You must specify at
      least one of ``df`` and ``knots``.
    :arg degree: The degree of the spline to use.
    :arg include_intercept: If ``True``, then the resulting
      spline basis will span the intercept term (i.e., the constant
      function). If ``False`` (the default) then this will not be the case,
      which is useful for avoiding overspecification in models that include
      multiple spline terms and/or an intercept term.
    :arg lower_bound: The lower exterior knot location.
    :arg upper_bound: The upper exterior knot location.

    A spline with ``degree=0`` is piecewise constant with breakpoints at each
    knot, and the default knot positions are quantiles of the input. So if you
    find yourself in the situation of wanting to quantize a continuous
    variable into ``num_bins`` equal-sized bins with a constant effect across
    each bin, you can use ``bs(x, num_bins - 1, degree=0)``. (The ``- 1`` is
    because one degree of freedom will be taken by the intercept;
    alternatively, you could leave the intercept term out of your model and
    use ``bs(x, num_bins, degree=0, include_intercept=True)``.

    A spline with ``degree=1`` is piecewise linear with breakpoints at each
    knot.

    The default is ``degree=3``, which gives a cubic b-spline.

    This is a stateful transform (for details see
    :ref:`stateful-transforms`). If ``knots``, ``lower_bound``, or
    ``upper_bound`` are not specified, they will be calculated from the data
    and then the chosen values will be remembered and re-used for prediction
    from the fitted model.

    Using this function requires scipy be installed.

    .. note:: This function is very similar to the R function of the same
      name. In cases where both return output at all (e.g., R's ``bs`` will
      raise an error if ``degree=0``, while patsy's will not), they should
      produce identical output given identical input and parameter settings.

    .. warning:: I'm not sure on what the proper handling of points outside
      the lower/upper bounds is, so for now attempting to evaluate a spline
      basis at such points produces an error. Patches gratefully accepted.

    .. versionadded:: 0.2.0
    