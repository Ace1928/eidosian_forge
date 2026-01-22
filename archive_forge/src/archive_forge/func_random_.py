import random
from yaql.language import specs
from yaql.language import yaqltypes
def random_():
    """:yaql:random

    Returns the next random floating number from [0.0, 1.0).

    :signature: random()
    :returnType: float

    .. code::

        yaql> random()
        0.6039529924951869
    """
    return random.random()