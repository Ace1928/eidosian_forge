from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
import numpy as np
import gym.error
from gym import logger
from gym.spaces.space import Space
def is_bounded(self, manner: str='both') -> bool:
    """Checks whether the box is bounded in some sense.

        Args:
            manner (str): One of ``"both"``, ``"below"``, ``"above"``.

        Returns:
            If the space is bounded

        Raises:
            ValueError: If `manner` is neither ``"both"`` nor ``"below"`` or ``"above"``
        """
    below = bool(np.all(self.bounded_below))
    above = bool(np.all(self.bounded_above))
    if manner == 'both':
        return below and above
    elif manner == 'below':
        return below
    elif manner == 'above':
        return above
    else:
        raise ValueError(f"manner is not in {{'below', 'above', 'both'}}, actual value: {manner}")