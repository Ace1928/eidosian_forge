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
def create_hit(self, hit_type=None, question=None, hit_layout=None, lifetime=datetime.timedelta(days=7), max_assignments=1, title=None, description=None, keywords=None, reward=None, duration=datetime.timedelta(days=7), approval_delay=None, annotation=None, questions=None, qualifications=None, layout_params=None, response_groups=None):
    """
        Creates a new HIT.
        Returns a ResultSet
        See: http://docs.amazonwebservices.com/AWSMechTurk/2012-03-25/AWSMturkAPI/ApiReference_CreateHITOperation.html
        """
    params = {'LifetimeInSeconds': self.duration_as_seconds(lifetime), 'MaxAssignments': max_assignments}
    neither = question is None and questions is None
    if hit_layout is None:
        both = question is not None and questions is not None
        if neither or both:
            raise ValueError('Must specify question (single Question instance) or questions (list or QuestionForm instance), but not both')
        if question:
            questions = [question]
        question_param = QuestionForm(questions)
        if isinstance(question, QuestionForm):
            question_param = question
        elif isinstance(question, ExternalQuestion):
            question_param = question
        elif isinstance(question, HTMLQuestion):
            question_param = question
        params['Question'] = question_param.get_as_xml()
    else:
        if not neither:
            raise ValueError('Must not specify question (single Question instance) or questions (list or QuestionForm instance) when specifying hit_layout')
        params['HITLayoutId'] = hit_layout
        if layout_params:
            params.update(layout_params.get_as_params())
    if hit_type:
        params['HITTypeId'] = hit_type
    else:
        final_keywords = MTurkConnection.get_keywords_as_string(keywords)
        final_price = MTurkConnection.get_price_as_price(reward)
        final_duration = self.duration_as_seconds(duration)
        additional_params = dict(Title=title, Description=description, Keywords=final_keywords, AssignmentDurationInSeconds=final_duration)
        additional_params.update(final_price.get_as_params('Reward'))
        if approval_delay is not None:
            d = self.duration_as_seconds(approval_delay)
            additional_params['AutoApprovalDelayInSeconds'] = d
        params.update(additional_params)
    if annotation is not None:
        params['RequesterAnnotation'] = annotation
    if qualifications is not None:
        params.update(qualifications.get_as_params())
    if response_groups:
        self.build_list_params(params, response_groups, 'ResponseGroup')
    return self._process_request('CreateHIT', params, [('HIT', HIT)])