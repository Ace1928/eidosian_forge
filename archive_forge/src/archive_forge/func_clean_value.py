def clean_value(self, key_name, value):
    """Clean the specified value and return it.

        If the value is not specified to be dealt with, the original value
        will be returned.
        """
    if key_name in self._to_process:
        try:
            cleaner = getattr(self, f'_clean_{key_name}')
        except AttributeError:
            raise AssertionError(f'No function to clean key: {key_name}') from None
        value = cleaner(value)
    return value