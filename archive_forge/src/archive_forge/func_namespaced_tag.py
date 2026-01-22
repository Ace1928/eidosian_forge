import io
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
import wandb.util
from wandb.sdk.lib import telemetry
from wandb.viz import custom_chart
def namespaced_tag(tag: str, namespace: str='') -> str:
    if not namespace:
        return tag
    elif tag in namespace:
        return namespace
    else:
        return namespace + '/' + tag