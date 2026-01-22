def ranks_from_sequence(seq):
    """Given a sequence, yields each element with an increasing rank, suitable
    for use as an argument to ``spearman_correlation``.
    """
    return ((k, i) for i, k in enumerate(seq))