import copy
from typing import Any, Dict
import stanio
@property
def stan_vars(self) -> Dict[str, stanio.Variable]:
    """
        These are the user-defined variables in the Stan program.
        """
    return self._stan_vars