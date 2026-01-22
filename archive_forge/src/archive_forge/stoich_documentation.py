import numpy as np
from chempy.units import unit_of, to_unitless
Decomposes yields into mass-action reactions

    This function offers a way to express a reaction with non-integer
    stoichiometric coefficients as a linear combination of production reactions
    with integer coefficients.

    Ak = y

    A is (n_species x n_reactions) matrix, k is "rate coefficient", y is yields


    Parameters
    ----------
    yields : OrderedDict
        Specie names as keys and yields as values.
    rxns : iterable :class:`Reaction` instances
        Dict keys must match those of ``yields`` each pair
        of dictionaries gives stoichiometry
        (1st is reactant, 2nd is products).
    atol : float
        Absolute tolerance for residuals.


    Examples
    --------
    >>> from chempy import Reaction
    >>> h2a = Reaction({'H2O': 1}, {'H2': 1, 'O': 1})
    >>> h2b = Reaction({'H2O': 1}, {'H2': 1, 'H2O2': 1}, inact_reac={'H2O': 1})
    >>> decompose_yields({'H2': 3, 'O': 2, 'H2O2': 1}, [h2a, h2b])
    array([2., 1.])

    Raises
    ------
    ValueError
        When atol is exceeded
    numpy.LinAlgError
        When numpy.linalg.lstsq fails to converge

    Returns
    -------
    1-dimensional array of effective rate coefficients.

    