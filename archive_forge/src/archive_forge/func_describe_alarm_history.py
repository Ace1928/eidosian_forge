from boto.compat import json, map, six, zip
from boto.connection import AWSQueryConnection
from boto.ec2.cloudwatch.metric import Metric
from boto.ec2.cloudwatch.alarm import MetricAlarm, MetricAlarms, AlarmHistoryItem
from boto.ec2.cloudwatch.datapoint import Datapoint
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
import boto
def describe_alarm_history(self, alarm_name=None, start_date=None, end_date=None, max_records=None, history_item_type=None, next_token=None):
    """
        Retrieves history for the specified alarm. Filter alarms by date range
        or item type. If an alarm name is not specified, Amazon CloudWatch
        returns histories for all of the owner's alarms.

        Amazon CloudWatch retains the history of deleted alarms for a period of
        six weeks. If an alarm has been deleted, its history can still be
        queried.

        :type alarm_name: string
        :param alarm_name: The name of the alarm.

        :type start_date: datetime
        :param start_date: The starting date to retrieve alarm history.

        :type end_date: datetime
        :param end_date: The starting date to retrieve alarm history.

        :type history_item_type: string
        :param history_item_type: The type of alarm histories to retreive
            (ConfigurationUpdate | StateUpdate | Action)

        :type max_records: int
        :param max_records: The maximum number of alarm descriptions
            to retrieve.

        :type next_token: string
        :param next_token: The token returned by a previous call to indicate
            that there is more data.

        :rtype list
        """
    params = {}
    if alarm_name:
        params['AlarmName'] = alarm_name
    if start_date:
        params['StartDate'] = start_date.isoformat()
    if end_date:
        params['EndDate'] = end_date.isoformat()
    if history_item_type:
        params['HistoryItemType'] = history_item_type
    if max_records:
        params['MaxRecords'] = max_records
    if next_token:
        params['NextToken'] = next_token
    return self.get_list('DescribeAlarmHistory', params, [('member', AlarmHistoryItem)])