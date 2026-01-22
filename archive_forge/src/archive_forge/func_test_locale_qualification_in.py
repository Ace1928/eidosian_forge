from tests.unit import AWSMockServiceTestCase
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import \
def test_locale_qualification_in(self):
    self.set_http_response(status_code=200, body=MOCK_SERVER_RESPONSE)
    q = ExternalQuestion(external_url='http://samplesite', frame_height=800)
    keywords = ['boto', 'test', 'doctest']
    title = 'Boto External Question Test'
    annotation = 'An annotation from boto external question test'
    qualifications = Qualifications()
    test_requirement = LocaleRequirement(comparator='In', locale=[('US', 'WA'), 'CA'])
    qualifications.add(test_requirement)
    create_hit_rs = self.service_connection.create_hit(question=q, lifetime=60 * 65, max_assignments=2, title=title, keywords=keywords, reward=0.05, duration=60 * 6, approval_delay=60 * 60, annotation=annotation, qualifications=qualifications)
    self.assert_request_parameters({'QualificationRequirement.1.Comparator': 'In', 'QualificationRequirement.1.LocaleValue.1.Country': 'US', 'QualificationRequirement.1.LocaleValue.1.Subdivision': 'WA', 'QualificationRequirement.1.LocaleValue.2.Country': 'CA', 'QualificationRequirement.1.QualificationTypeId': '00000000000000000071'}, ignore_params_values=['AWSAccessKeyId', 'SignatureVersion', 'Timestamp', 'Title', 'Question', 'AssignmentDurationInSeconds', 'RequesterAnnotation', 'Version', 'LifetimeInSeconds', 'AutoApprovalDelayInSeconds', 'Reward.1.Amount', 'Description', 'MaxAssignments', 'Reward.1.CurrencyCode', 'Keywords', 'Operation'])
    self.assertEquals(create_hit_rs.status, True)