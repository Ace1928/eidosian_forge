from typing import Any, Dict, List, Optional
from tune.concepts.space import to_template
from tune.concepts.space.parameters import TuningParametersTemplate
@property
def dfs(self) -> Dict[str, Any]:
    """Dataframes extracted from
        :class:`~.tune.concepts.dataset.TuneDataset`
        """
    return self._dfs