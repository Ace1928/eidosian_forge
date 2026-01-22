from typing import Dict, Union, Optional, TYPE_CHECKING
from ray.util.annotations import PublicAPI
Label based node affinity scheduling strategy

    scheduling_strategy=NodeLabelSchedulingStrategy({
          "region": In("us"),
          "gpu_type": Exists()
    })
    