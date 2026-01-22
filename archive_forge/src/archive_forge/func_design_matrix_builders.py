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
def design_matrix_builders(termlists, data_iter_maker, eval_env, NA_action='drop'):
    """Construct several :class:`DesignInfo` objects from termlists.

    This is one of Patsy's fundamental functions. This function and
    :func:`build_design_matrices` together form the API to the core formula
    interpretation machinery.

    :arg termlists: A list of termlists, where each termlist is a list of
      :class:`Term` objects which together specify a design matrix.
    :arg data_iter_maker: A zero-argument callable which returns an iterator
      over dict-like data objects. This must be a callable rather than a
      simple iterator because sufficiently complex formulas may require
      multiple passes over the data (e.g. if there are nested stateful
      transforms).
    :arg eval_env: Either a :class:`EvalEnvironment` which will be used to
      look up any variables referenced in `termlists` that cannot be
      found in `data_iter_maker`, or else a depth represented as an
      integer which will be passed to :meth:`EvalEnvironment.capture`.
      ``eval_env=0`` means to use the context of the function calling
      :func:`design_matrix_builders` for lookups. If calling this function
      from a library, you probably want ``eval_env=1``, which means that
      variables should be resolved in *your* caller's namespace.
    :arg NA_action: An :class:`NAAction` object or string, used to determine
      what values count as 'missing' for purposes of determining the levels of
      categorical factors.
    :returns: A list of :class:`DesignInfo` objects, one for each
      termlist passed in.

    This function performs zero or more iterations over the data in order to
    sniff out any necessary information about factor types, set up stateful
    transforms, pick column names, etc.

    See :ref:`formulas` for details.

    .. versionadded:: 0.2.0
       The ``NA_action`` argument.
    .. versionadded:: 0.4.0
       The ``eval_env`` argument.
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)
    if isinstance(NA_action, str):
        NA_action = NAAction(NA_action)
    all_factors = set()
    for termlist in termlists:
        for term in termlist:
            all_factors.update(term.factors)
    factor_states = _factors_memorize(all_factors, data_iter_maker, eval_env)
    num_column_counts, cat_levels_contrasts = _examine_factor_types(all_factors, factor_states, data_iter_maker, NA_action)
    factor_infos = {}
    for factor in all_factors:
        if factor in num_column_counts:
            fi = FactorInfo(factor, 'numerical', factor_states[factor], num_columns=num_column_counts[factor], categories=None)
        else:
            assert factor in cat_levels_contrasts
            categories = cat_levels_contrasts[factor][0]
            fi = FactorInfo(factor, 'categorical', factor_states[factor], num_columns=None, categories=categories)
        factor_infos[factor] = fi
    design_infos = []
    for termlist in termlists:
        term_to_subterm_infos = _make_subterm_infos(termlist, num_column_counts, cat_levels_contrasts)
        assert isinstance(term_to_subterm_infos, OrderedDict)
        assert frozenset(term_to_subterm_infos) == frozenset(termlist)
        this_design_factor_infos = {}
        for term in termlist:
            for factor in term.factors:
                this_design_factor_infos[factor] = factor_infos[factor]
        column_names = []
        for subterms in six.itervalues(term_to_subterm_infos):
            for subterm in subterms:
                for column_name in _subterm_column_names_iter(factor_infos, subterm):
                    column_names.append(column_name)
        design_infos.append(DesignInfo(column_names, factor_infos=this_design_factor_infos, term_codings=term_to_subterm_infos))
    return design_infos