import datetime
from boto.ec2.cloudwatch import CloudWatchConnection
from tests.compat import unittest, OrderedDict
def test_describe_alarms(self):
    c = CloudWatchConnection()

    def make_request(*args, **kwargs):

        class Body(object):

            def __init__(self):
                self.status = 200

            def read(self):
                return DESCRIBE_ALARMS_BODY
        return Body()
    c.make_request = make_request
    alarms = c.describe_alarms()
    self.assertEquals(alarms.next_token, 'mynexttoken')
    self.assertEquals(alarms[0].name, 'FancyAlarm')
    self.assertEquals(alarms[0].comparison, '<')
    self.assertEquals(alarms[0].dimensions, {u'Job': [u'ANiceCronJob']})
    self.assertEquals(alarms[1].name, 'SuperFancyAlarm')
    self.assertEquals(alarms[1].comparison, '>')
    self.assertEquals(alarms[1].dimensions, {u'Job': [u'ABadCronJob']})