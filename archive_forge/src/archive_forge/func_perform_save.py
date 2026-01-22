import json
import pathlib
import warnings
from typing import IO, Union, Optional, Literal
from .mimebundle import spec_to_mimebundle
from ..vegalite.v5.data import data_transformers
from altair.utils._vegafusion_data import using_vegafusion
def perform_save():
    spec = chart.to_dict(context={'pre_transform': False})
    inner_mode = set_inspect_mode_argument(mode, embed_options or {}, spec, vegalite_version)
    if format == 'json':
        json_spec = json.dumps(spec, **json_kwds)
        write_file_or_filename(fp, json_spec, mode='w', encoding=kwargs.get('encoding', 'utf-8'))
    elif format == 'html':
        if inline:
            kwargs['template'] = 'inline'
        mimebundle = spec_to_mimebundle(spec=spec, format=format, mode=inner_mode, vega_version=vega_version, vegalite_version=vegalite_version, vegaembed_version=vegaembed_version, embed_options=embed_options, json_kwds=json_kwds, **kwargs)
        write_file_or_filename(fp, mimebundle['text/html'], mode='w', encoding=kwargs.get('encoding', 'utf-8'))
    elif format in ['png', 'svg', 'pdf', 'vega']:
        mimebundle = spec_to_mimebundle(spec=spec, format=format, mode=inner_mode, vega_version=vega_version, vegalite_version=vegalite_version, vegaembed_version=vegaembed_version, embed_options=embed_options, webdriver=webdriver, scale_factor=scale_factor, engine=engine, **kwargs)
        if format == 'png':
            write_file_or_filename(fp, mimebundle[0]['image/png'], mode='wb')
        elif format == 'pdf':
            write_file_or_filename(fp, mimebundle['application/pdf'], mode='wb')
        else:
            encoding = kwargs.get('encoding', 'utf-8')
            write_file_or_filename(fp, mimebundle['image/svg+xml'], mode='w', encoding=encoding)
    else:
        raise ValueError("Unsupported format: '{}'".format(format))