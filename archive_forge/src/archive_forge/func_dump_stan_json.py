from typing import Any, Mapping
import numpy as np
def dump_stan_json(data: Mapping[str, Any]) -> str:
    """
    Convert a mapping of strings to data to a JSON string.

    Values can be any numeric type, a boolean (converted to int),
    or any collection compatible with :func:`numpy.asarray`, e.g a
    :class:`pandas.Series`.

    Produces a string compatible with the
    `Json Format for Cmdstan
    <https://mc-stan.org/docs/cmdstan-guide/json.html>`__

    :param data: A mapping from strings to values. This can be a dictionary
        or something more exotic like an :class:`xarray.Dataset`. This will be
        copied before type conversion, not modified
    """
    return json.dumps(process_dictionary(data))