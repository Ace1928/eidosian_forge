import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def build_design_matrices(design_infos, data, NA_action='drop', return_type='matrix', dtype=np.dtype(float)):
    """Construct several design matrices from :class:`DesignMatrixBuilder`
    objects.

    This is one of Patsy's fundamental functions. This function and
    :func:`design_matrix_builders` together form the API to the core formula
    interpretation machinery.

    :arg design_infos: A list of :class:`DesignInfo` objects describing the
      design matrices to be built.
    :arg data: A dict-like object which will be used to look up data.
    :arg NA_action: What to do with rows that contain missing values. You can
      ``"drop"`` them, ``"raise"`` an error, or for customization, pass an
      :class:`NAAction` object. See :class:`NAAction` for details on what
      values count as 'missing' (and how to alter this).
    :arg return_type: Either ``"matrix"`` or ``"dataframe"``. See below.
    :arg dtype: The dtype of the returned matrix. Useful if you want to use
      single-precision or extended-precision.

    This function returns either a list of :class:`DesignMatrix` objects (for
    ``return_type="matrix"``) or a list of :class:`pandas.DataFrame` objects
    (for ``return_type="dataframe"``). In both cases, all returned design
    matrices will have ``.design_info`` attributes containing the appropriate
    :class:`DesignInfo` objects.

    Note that unlike :func:`design_matrix_builders`, this function takes only
    a simple data argument, not any kind of iterator. That's because this
    function doesn't need a global view of the data -- everything that depends
    on the whole data set is already encapsulated in the ``design_infos``. If
    you are incrementally processing a large data set, simply call this
    function for each chunk.

    Index handling: This function always checks for indexes in the following
    places:

    * If ``data`` is a :class:`pandas.DataFrame`, its ``.index`` attribute.
    * If any factors evaluate to a :class:`pandas.Series` or
      :class:`pandas.DataFrame`, then their ``.index`` attributes.

    If multiple indexes are found, they must be identical (same values in the
    same order). If no indexes are found, then a default index is generated
    using ``np.arange(num_rows)``. One way or another, we end up with a single
    index for all the data. If ``return_type="dataframe"``, then this index is
    used as the index of the returned DataFrame objects. Examining this index
    makes it possible to determine which rows were removed due to NAs.

    Determining the number of rows in design matrices: This is not as obvious
    as it might seem, because it's possible to have a formula like "~ 1" that
    doesn't depend on the data (it has no factors). For this formula, it's
    obvious what every row in the design matrix should look like (just the
    value ``1``); but, how many rows like this should there be? To determine
    the number of rows in a design matrix, this function always checks in the
    following places:

    * If ``data`` is a :class:`pandas.DataFrame`, then its number of rows.
    * The number of entries in any factors present in any of the design
    * matrices being built.

    All these values much match. In particular, if this function is called to
    generate multiple design matrices at once, then they must all have the
    same number of rows.

    .. versionadded:: 0.2.0
       The ``NA_action`` argument.

    """
    if isinstance(NA_action, str):
        NA_action = NAAction(NA_action)
    if return_type == 'dataframe' and (not have_pandas):
        raise PatsyError('pandas.DataFrame was requested, but pandas is not installed')
    if return_type not in ('matrix', 'dataframe'):
        raise PatsyError("unrecognized output type %r, should be 'matrix' or 'dataframe'" % (return_type,))
    factor_info_to_values = {}
    factor_info_to_isNAs = {}
    rows_checker = _CheckMatch('Number of rows', lambda a, b: a == b)
    index_checker = _CheckMatch('Index', lambda a, b: a.equals(b))
    if have_pandas and isinstance(data, pandas.DataFrame):
        index_checker.check(data.index, 'data.index', None)
        rows_checker.check(data.shape[0], 'data argument', None)
    for design_info in design_infos:
        for factor_info in six.itervalues(design_info.factor_infos):
            if factor_info not in factor_info_to_values:
                value, is_NA = _eval_factor(factor_info, data, NA_action)
                factor_info_to_isNAs[factor_info] = is_NA
                name = factor_info.factor.name()
                origin = factor_info.factor.origin
                rows_checker.check(value.shape[0], name, origin)
                if have_pandas and isinstance(value, (pandas.Series, pandas.DataFrame)):
                    index_checker.check(value.index, name, origin)
                value = np.asarray(value)
                factor_info_to_values[factor_info] = value
    values = list(factor_info_to_values.values())
    is_NAs = list(factor_info_to_isNAs.values())
    origins = [factor_info.factor.origin for factor_info in factor_info_to_values]
    pandas_index = index_checker.value
    num_rows = rows_checker.value
    if return_type == 'dataframe' and num_rows is not None:
        if pandas_index is None:
            pandas_index = np.arange(num_rows)
        values.append(pandas_index)
        is_NAs.append(np.zeros(len(pandas_index), dtype=bool))
        origins.append(None)
    new_values = NA_action.handle_NA(values, is_NAs, origins)
    if new_values:
        num_rows = new_values[0].shape[0]
    if return_type == 'dataframe' and num_rows is not None:
        pandas_index = new_values.pop()
    factor_info_to_values = dict(zip(factor_info_to_values, new_values))
    results = []
    for design_info in design_infos:
        results.append(_build_design_matrix(design_info, factor_info_to_values, dtype))
    matrices = []
    for need_reshape, matrix in results:
        if need_reshape:
            assert matrix.shape[0] == 1
            if num_rows is not None:
                matrix = DesignMatrix(np.repeat(matrix, num_rows, axis=0), matrix.design_info)
            else:
                raise PatsyError("No design matrix has any non-trivial factors, the data object is not a DataFrame. I can't tell how many rows the design matrix should have!")
        matrices.append(matrix)
    if return_type == 'dataframe':
        assert have_pandas
        for i, matrix in enumerate(matrices):
            di = matrix.design_info
            matrices[i] = pandas.DataFrame(matrix, columns=di.column_names, index=pandas_index)
            matrices[i].design_info = di
    return matrices