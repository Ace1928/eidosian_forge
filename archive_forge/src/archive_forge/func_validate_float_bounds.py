from __future__ import annotations
import numbers
@classmethod
def validate_float_bounds(cls, value: int | float, value_name: str | None) -> None:
    """Validate that a float value can be represented by a JavaScript Number.

        Parameters
        ----------
        value : float
        value_name : str or None
            The name of the value parameter. If specified, this will be used
            in any exception that is thrown.

        Raises
        ------
        JSNumberBoundsException
            Raised with a human-readable explanation if the value falls outside
            JavaScript float bounds.

        """
    if value_name is None:
        value_name = 'value'
    if not isinstance(value, (numbers.Integral, float)):
        raise JSNumberBoundsException(f'{value_name} ({value}) is not a float')
    elif value < cls.MIN_NEGATIVE_VALUE:
        raise JSNumberBoundsException(f'{value_name} ({value}) must be >= -1.797e+308')
    elif value > cls.MAX_VALUE:
        raise JSNumberBoundsException(f'{value_name} ({value}) must be <= 1.797e+308')