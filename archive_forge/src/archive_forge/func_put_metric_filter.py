import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def put_metric_filter(self, log_group_name, filter_name, filter_pattern, metric_transformations):
    """
        Creates or updates a metric filter and associates it with the
        specified log group. Metric filters allow you to configure
        rules to extract metric data from log events ingested through
        `PutLogEvents` requests.

        :type log_group_name: string
        :param log_group_name:

        :type filter_name: string
        :param filter_name: The name of the metric filter.

        :type filter_pattern: string
        :param filter_pattern:

        :type metric_transformations: list
        :param metric_transformations:

        """
    params = {'logGroupName': log_group_name, 'filterName': filter_name, 'filterPattern': filter_pattern, 'metricTransformations': metric_transformations}
    return self.make_request(action='PutMetricFilter', body=json.dumps(params))