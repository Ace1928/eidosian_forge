def initIterate(container, generator):
    """Call the function *container* with an iterable as
    its only argument. The iterable must be returned by
    the method or the object *generator*.

    :param container: The type to put in the data from func.
    :param generator: A function returning an iterable (list, tuple, ...),
                      the content of this iterable will fill the container.
    :returns: An instance of the container filled with data from the
              generator.

    This helper function can be used in conjunction with a Toolbox
    to register a generator of filled containers, as individuals or
    population.

        >>> import random
        >>> from functools import partial
        >>> random.seed(42)
        >>> gen_idx = partial(random.sample, range(10), 10)
        >>> initIterate(list, gen_idx)      # doctest: +SKIP
        [1, 0, 4, 9, 6, 5, 8, 2, 3, 7]

    See the :ref:`permutation` and :ref:`arithmetic-expr` tutorials for
    more examples.
    """
    return container(generator())