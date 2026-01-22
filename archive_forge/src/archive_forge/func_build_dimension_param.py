from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def build_dimension_param(self, dimension, params):
    prefix = 'Dimensions.member'
    i = 0
    for dim_name in dimension:
        dim_value = dimension[dim_name]
        if dim_value:
            if isinstance(dim_value, six.string_types):
                dim_value = [dim_value]
            for value in dim_value:
                params['%s.%d.Name' % (prefix, i + 1)] = dim_name
                params['%s.%d.Value' % (prefix, i + 1)] = value
                i += 1
        else:
            params['%s.%d.Name' % (prefix, i + 1)] = dim_name
            i += 1