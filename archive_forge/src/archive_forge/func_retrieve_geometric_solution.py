def retrieve_geometric_solution(M, N=2, numerical=False, prefer_rur=False, data_url=None, verbose=True):
    """
    Given a manifold M, retrieve the exact or numerical solutions to the
    Ptolemy variety (pass numerical = True for numerical solutions). The additional
    options are the same as the ones passed to retrieve_solutions of a PtolemyVariety.

    >>> from snappy.ptolemy.geometricRep import retrieve_geometric_solutions
    >>> retrieve_geometric_solutions(Manifold("m004")) #doctest: +SKIP

    """
    return compute_geometric_solution(M, N, numerical=numerical, engine='retrieve', prefer_rur=prefer_rur, data_url=data_url, verbose=verbose)