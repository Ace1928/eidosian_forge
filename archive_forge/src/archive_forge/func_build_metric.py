import io as StringIO
import math
import re
from ..metrics_core import Metric, METRIC_LABEL_NAME_RE
from ..samples import Exemplar, Sample, Timestamp
from ..utils import floatToGoString
def build_metric(name, documentation, typ, unit, samples):
    if typ is None:
        typ = 'unknown'
    for suffix in set(type_suffixes.get(typ, []) + ['']):
        if name + suffix in seen_names:
            raise ValueError('Clashing name: ' + name + suffix)
        seen_names.add(name + suffix)
    if documentation is None:
        documentation = ''
    if unit is None:
        unit = ''
    if unit and (not name.endswith('_' + unit)):
        raise ValueError('Unit does not match metric name: ' + name)
    if unit and typ in ['info', 'stateset']:
        raise ValueError('Units not allowed for this metric type: ' + name)
    if typ in ['histogram', 'gaugehistogram']:
        _check_histogram(samples, name)
    metric = Metric(name, documentation, typ, unit)
    metric.samples = samples
    return metric