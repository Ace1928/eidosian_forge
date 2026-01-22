from typing import List, Optional, Tuple, Dict, Iterable, overload, Union
from altair import (
from altair.utils._vegafusion_data import get_inline_tables, import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.schemapi import Undefined
def name_views(chart: Union[Chart, FacetChart, LayerChart, HConcatChart, VConcatChart, ConcatChart], i: int=0, exclude: Optional[Iterable[str]]=None) -> List[str]:
    """Name unnamed chart views

    Name unnamed charts views so that we can look them up later in
    the compiled Vega spec.

    Note: This function mutates the input chart by applying names to
    unnamed views.

    Parameters
    ----------
    chart : Chart, FacetChart, LayerChart, HConcatChart, VConcatChart, or ConcatChart
        Altair chart to apply names to
    i : int (default 0)
        Starting chart index
    exclude : iterable of str
        Names of charts to exclude

    Returns
    -------
    list of str
        List of the names of the charts and subcharts
    """
    exclude = set(exclude) if exclude is not None else set()
    if isinstance(chart, _chart_class_mapping[Chart]) or isinstance(chart, _chart_class_mapping[FacetChart]):
        if chart.name not in exclude:
            if chart.name in (None, Undefined):
                chart.name = Chart._get_name()
            return [chart.name]
        else:
            return []
    else:
        if isinstance(chart, _chart_class_mapping[LayerChart]):
            subcharts = chart.layer
        elif isinstance(chart, _chart_class_mapping[HConcatChart]):
            subcharts = chart.hconcat
        elif isinstance(chart, _chart_class_mapping[VConcatChart]):
            subcharts = chart.vconcat
        elif isinstance(chart, _chart_class_mapping[ConcatChart]):
            subcharts = chart.concat
        else:
            raise ValueError(f'transformed_data accepts an instance of Chart, FacetChart, LayerChart, HConcatChart, VConcatChart, or ConcatChart\nReceived value of type: {type(chart)}')
        chart_names: List[str] = []
        for subchart in subcharts:
            for name in name_views(subchart, i=i + len(chart_names), exclude=exclude):
                chart_names.append(name)
        return chart_names