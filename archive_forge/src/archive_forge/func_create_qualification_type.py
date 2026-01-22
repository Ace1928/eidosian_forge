import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
def create_qualification_type(self, name, description, status, keywords=None, retry_delay=None, test=None, answer_key=None, answer_key_xml=None, test_duration=None, auto_granted=False, auto_granted_value=1):
    """
        Create a new Qualification Type.

        name: This will be visible to workers and must be unique for a
           given requester.

        description: description shown to workers.  Max 2000 characters.

        status: 'Active' or 'Inactive'

        keywords: list of keyword strings or comma separated string.
           Max length of 1000 characters when concatenated with commas.

        retry_delay: number of seconds after requesting a
           qualification the worker must wait before they can ask again.
           If not specified, workers can only request this qualification
           once.

        test: a QuestionForm

        answer_key: an XML string of your answer key, for automatically
           scored qualification tests.
           (Consider implementing an AnswerKey class for this to support.)

        test_duration: the number of seconds a worker has to complete the test.

        auto_granted: if True, requests for the Qualification are granted
           immediately.  Can't coexist with a test.

        auto_granted_value: auto_granted qualifications are given this value.

        """
    params = {'Name': name, 'Description': description, 'QualificationTypeStatus': status}
    if retry_delay is not None:
        params['RetryDelayInSeconds'] = retry_delay
    if test is not None:
        assert isinstance(test, QuestionForm)
        assert test_duration is not None
        params['Test'] = test.get_as_xml()
    if test_duration is not None:
        params['TestDurationInSeconds'] = test_duration
    if answer_key is not None:
        if isinstance(answer_key, basestring):
            params['AnswerKey'] = answer_key
        else:
            raise TypeError
    if auto_granted:
        assert test is None
        params['AutoGranted'] = True
        params['AutoGrantedValue'] = auto_granted_value
    if keywords:
        params['Keywords'] = self.get_keywords_as_string(keywords)
    return self._process_request('CreateQualificationType', params, [('QualificationType', QualificationType)])