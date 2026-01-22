from typing import Optional
from ...public import PanelMetricsHelper, Run
from .runset import Runset
from .util import Attr, Base, Panel, nested_get, nested_set
@classmethod
def from_panel_agg(cls, runset: 'Runset', panel: 'Panel', metric: str) -> 'LineKey':
    key = f'{runset.id}-config:group:{panel.groupby}:null:{metric}'
    return cls(key)