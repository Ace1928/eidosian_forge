from typing import Optional
from ...public import PanelMetricsHelper, Run
from .runset import Runset
from .util import Attr, Base, Panel, nested_get, nested_set
@classmethod
def from_runset_agg(cls, runset: 'Runset', metric: str) -> 'LineKey':
    groupby = runset.groupby
    if runset.groupby is None:
        groupby = 'null'
    key = f'{runset.id}-run:group:{groupby}:{metric}'
    return cls(key)