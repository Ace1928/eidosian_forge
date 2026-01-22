from argparse import ArgumentParser
from argparse import ArgumentTypeError
from unittest import mock
from datetime import datetime
from testtools import ExpectedException
from vitrageclient.tests.cli.base import CliTestCase
from vitrageclient.v1.cli.event import EventPost
@mock.patch('vitrageclient.v1.cli.event.datetime')
def test_parser_event_post_without_time_uses_time_now(self, dt_mock):
    current_time = datetime.now()
    dt_mock.now.return_value = current_time
    iso_time = current_time.isoformat()
    parser = self.event_post.get_parser('vitrage event post')
    args = parser.parse_args(args=['--type', 'bla', '--details', '{"blabla":[]}'])
    with mock.patch.object(self.app.client.event, 'post') as poster_mock:
        self.event_post.take_action(args)
        poster_mock.assert_called_with(event_time=iso_time, details={'blabla': []}, event_type='bla')