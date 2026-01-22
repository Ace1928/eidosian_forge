def to_call(self, the_callable, *args, **kwargs):
    """
        Sets up the callable & any arguments to run it with.

        This is stored for subsequent calls so that those queries can be
        run without requiring user intervention.

        Example::

            # Just an example callable.
            >>> def squares_to(y):
            ...     for x in range(1, y):
            ...         yield x**2
            >>> rs = ResultSet()
            # Set up what to call & arguments.
            >>> rs.to_call(squares_to, y=3)

        """
    if not callable(the_callable):
        raise ValueError('You must supply an object or function to be called.')
    self._limit = kwargs.pop('limit', None)
    if self._limit is not None and self._limit < 0:
        self._limit = None
    self.the_callable = the_callable
    self.call_args = args
    self.call_kwargs = kwargs