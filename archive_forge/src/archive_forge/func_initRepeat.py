def initRepeat(container, func, n):
    """Call the function *func* *n* times and return the results in a
    container type `container`

    :param container: The type to put in the data from func.
    :param func: The function that will be called n times to fill the
                 container.
    :param n: The number of times to repeat func.
    :returns: An instance of the container filled with data from func.

    This helper function can be used in conjunction with a Toolbox
    to register a generator of filled containers, as individuals or
    population.

        >>> import random
        >>> random.seed(42)
        >>> initRepeat(list, random.random, 2) # doctest: +ELLIPSIS,
        ...                                    # doctest: +NORMALIZE_WHITESPACE
        [0.6394..., 0.0250...]

    See the :ref:`list-of-floats` and :ref:`population` tutorials for more examples.
    """
    return container((func() for _ in range(n)))