import json
import pathlib
import warnings
from typing import IO, Union, Optional, Literal
from .mimebundle import spec_to_mimebundle
from ..vegalite.v5.data import data_transformers
from altair.utils._vegafusion_data import using_vegafusion
def set_inspect_mode_argument(mode: Optional[Literal['vega-lite']], embed_options: dict, spec: dict, vegalite_version: Optional[str]) -> Literal['vega-lite']:
    """Inspect the mode argument in the save function"""
    if mode is None:
        if 'mode' in embed_options:
            mode = embed_options['mode']
        elif '$schema' in spec:
            mode = spec['$schema'].split('/')[-2]
        else:
            mode = 'vega-lite'
    if mode != 'vega-lite':
        raise ValueError("mode must be 'vega-lite', not '{}'".format(mode))
    if mode == 'vega-lite' and vegalite_version is None:
        raise ValueError('must specify vega-lite version')
    return mode