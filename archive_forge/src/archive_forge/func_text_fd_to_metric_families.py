import io as StringIO
import math
import re
from ..metrics_core import Metric, METRIC_LABEL_NAME_RE
from ..samples import Exemplar, Sample, Timestamp
from ..utils import floatToGoString
def text_fd_to_metric_families(fd):
    """Parse Prometheus text format from a file descriptor.

    This is a laxer parser than the main Go parser,
    so successful parsing does not imply that the parsed
    text meets the specification.

    Yields Metric's.
    """
    name = None
    allowed_names = []
    eof = False
    seen_names = set()
    type_suffixes = {'counter': ['_total', '_created'], 'summary': ['', '_count', '_sum', '_created'], 'histogram': ['_count', '_sum', '_bucket', '_created'], 'gaugehistogram': ['_gcount', '_gsum', '_bucket'], 'info': ['_info']}

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
    for line in fd:
        if line[-1] == '\n':
            line = line[:-1]
        if eof:
            raise ValueError('Received line after # EOF: ' + line)
        if not line:
            raise ValueError('Received blank line')
        if line == '# EOF':
            eof = True
        elif line.startswith('#'):
            parts = line.split(' ', 3)
            if len(parts) < 4:
                raise ValueError('Invalid line: ' + line)
            if parts[2] == name and samples:
                raise ValueError('Received metadata after samples: ' + line)
            if parts[2] != name:
                if name is not None:
                    yield build_metric(name, documentation, typ, unit, samples)
                name = parts[2]
                unit = None
                typ = None
                documentation = None
                group = None
                seen_groups = set()
                group_timestamp = None
                group_timestamp_samples = set()
                samples = []
                allowed_names = [parts[2]]
            if parts[1] == 'HELP':
                if documentation is not None:
                    raise ValueError('More than one HELP for metric: ' + line)
                documentation = _unescape_help(parts[3])
            elif parts[1] == 'TYPE':
                if typ is not None:
                    raise ValueError('More than one TYPE for metric: ' + line)
                typ = parts[3]
                if typ == 'untyped':
                    raise ValueError('Invalid TYPE for metric: ' + line)
                allowed_names = [name + n for n in type_suffixes.get(typ, [''])]
            elif parts[1] == 'UNIT':
                if unit is not None:
                    raise ValueError('More than one UNIT for metric: ' + line)
                unit = parts[3]
            else:
                raise ValueError('Invalid line: ' + line)
        else:
            sample = _parse_sample(line)
            if sample.name not in allowed_names:
                if name is not None:
                    yield build_metric(name, documentation, typ, unit, samples)
                name = sample.name
                documentation = None
                unit = None
                typ = 'unknown'
                samples = []
                group = None
                group_timestamp = None
                group_timestamp_samples = set()
                seen_groups = set()
                allowed_names = [sample.name]
            if typ == 'stateset' and name not in sample.labels:
                raise ValueError('Stateset missing label: ' + line)
            if name + '_bucket' == sample.name and (sample.labels.get('le', 'NaN') == 'NaN' or _isUncanonicalNumber(sample.labels['le'])):
                raise ValueError('Invalid le label: ' + line)
            if name + '_bucket' == sample.name and (not isinstance(sample.value, int) and (not sample.value.is_integer())):
                raise ValueError('Bucket value must be an integer: ' + line)
            if (name + '_count' == sample.name or name + '_gcount' == sample.name) and (not isinstance(sample.value, int) and (not sample.value.is_integer())):
                raise ValueError('Count value must be an integer: ' + line)
            if typ == 'summary' and name == sample.name and (not 0 <= float(sample.labels.get('quantile', -1)) <= 1 or _isUncanonicalNumber(sample.labels['quantile'])):
                raise ValueError('Invalid quantile label: ' + line)
            g = tuple(sorted(_group_for_sample(sample, name, typ).items()))
            if group is not None and g != group and (g in seen_groups):
                raise ValueError('Invalid metric grouping: ' + line)
            if group is not None and g == group:
                if (sample.timestamp is None) != (group_timestamp is None):
                    raise ValueError('Mix of timestamp presence within a group: ' + line)
                if group_timestamp is not None and group_timestamp > sample.timestamp and (typ != 'info'):
                    raise ValueError('Timestamps went backwards within a group: ' + line)
            else:
                group_timestamp_samples = set()
            series_id = (sample.name, tuple(sorted(sample.labels.items())))
            if sample.timestamp != group_timestamp or series_id not in group_timestamp_samples:
                samples.append(sample)
            group_timestamp_samples.add(series_id)
            group = g
            group_timestamp = sample.timestamp
            seen_groups.add(g)
            if typ == 'stateset' and sample.value not in [0, 1]:
                raise ValueError('Stateset samples can only have values zero and one: ' + line)
            if typ == 'info' and sample.value != 1:
                raise ValueError('Info samples can only have value one: ' + line)
            if typ == 'summary' and name == sample.name and (sample.value < 0):
                raise ValueError('Quantile values cannot be negative: ' + line)
            if sample.name[len(name):] in ['_total', '_sum', '_count', '_bucket', '_gcount', '_gsum'] and math.isnan(sample.value):
                raise ValueError('Counter-like samples cannot be NaN: ' + line)
            if sample.name[len(name):] in ['_total', '_sum', '_count', '_bucket', '_gcount'] and sample.value < 0:
                raise ValueError('Counter-like samples cannot be negative: ' + line)
            if sample.exemplar and (not (typ in ['histogram', 'gaugehistogram'] and sample.name.endswith('_bucket') or (typ in ['counter'] and sample.name.endswith('_total')))):
                raise ValueError('Invalid line only histogram/gaugehistogram buckets and counters can have exemplars: ' + line)
    if name is not None:
        yield build_metric(name, documentation, typ, unit, samples)
    if not eof:
        raise ValueError('Missing # EOF at end')