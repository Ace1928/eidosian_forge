from typing import Any, Dict, List, Optional
from tune.concepts.space import to_template
from tune.concepts.space.parameters import TuningParametersTemplate
def with_params(self, params: Any) -> 'Trial':
    """Set parameters for the trial, a new Trial object will
        be constructed and with the new ``params``

        :param params: parameters for tuning
        """
    t = self.copy()
    t._params = to_template(params)
    return t