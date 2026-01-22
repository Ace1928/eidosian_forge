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
def get_file_upload_url(self, assignment_id, question_identifier):
    """
        Generates and returns a temporary URL to an uploaded file. The
        temporary URL is used to retrieve the file as an answer to a
        FileUploadAnswer question, it is valid for 60 seconds.

        Will have a FileUploadURL attribute as per the API Reference.
        """
    params = {'AssignmentId': assignment_id, 'QuestionIdentifier': question_identifier}
    return self._process_request('GetFileUploadURL', params, [('FileUploadURL', FileUploadURL)])