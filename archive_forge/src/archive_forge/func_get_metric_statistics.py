from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def get_metric_statistics(self, period, start_time, end_time, metric_name, namespace, statistics, dimensions=None, unit=None):
    """
        Get time-series data for one or more statistics of a given metric.

        :type period: integer
        :param period: The granularity, in seconds, of the returned datapoints.
            Period must be at least 60 seconds and must be a multiple
            of 60. The default value is 60.

        :type start_time: datetime
        :param start_time: The time stamp to use for determining the
            first datapoint to return. The value specified is
            inclusive; results include datapoints with the time stamp
            specified.

        :type end_time: datetime
        :param end_time: The time stamp to use for determining the
            last datapoint to return. The value specified is
            exclusive; results will include datapoints up to the time
            stamp specified.

        :type metric_name: string
        :param metric_name: The metric name.

        :type namespace: string
        :param namespace: The metric's namespace.

        :type statistics: list
        :param statistics: A list of statistics names Valid values:
            Average | Sum | SampleCount | Maximum | Minimum

        :type dimensions: dict
        :param dimensions: A dictionary of dimension key/values where
                           the key is the dimension name and the value
                           is either a scalar value or an iterator
                           of values to be associated with that
                           dimension.

        :type unit: string
        :param unit: The unit for the metric.  Value values are:
            Seconds | Microseconds | Milliseconds | Bytes | Kilobytes |
            Megabytes | Gigabytes | Terabytes | Bits | Kilobits |
            Megabits | Gigabits | Terabits | Percent | Count |
            Bytes/Second | Kilobytes/Second | Megabytes/Second |
            Gigabytes/Second | Terabytes/Second | Bits/Second |
            Kilobits/Second | Megabits/Second | Gigabits/Second |
            Terabits/Second | Count/Second | None

        :rtype: list
        """
    params = {'Period': period, 'MetricName': metric_name, 'Namespace': namespace, 'StartTime': start_time.isoformat(), 'EndTime': end_time.isoformat()}
    self.build_list_params(params, statistics, 'Statistics.member.%d')
    if dimensions:
        self.build_dimension_param(dimensions, params)
    if unit:
        params['Unit'] = unit
    return self.get_list('GetMetricStatistics', params, [('member', Datapoint)])