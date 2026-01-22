import testtools
from unittest import mock
from aodhclient.v2 import alarm
def test_clean_rules_event_alarm(self):
    am = alarm.AlarmManager(self.client)
    alarm_value = self.alarms.get('event_alarm')
    am._clean_rules('event', alarm_value)
    alarm_value.pop('type')
    alarm_value.pop('name')
    result = self.results.get('result1')
    self.assertEqual(alarm_value, result)