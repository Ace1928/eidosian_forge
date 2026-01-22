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
def get_assignments(self, hit_id, status=None, sort_by='SubmitTime', sort_direction='Ascending', page_size=10, page_number=1, response_groups=None):
    """
        Retrieves completed assignments for a HIT.
        Use this operation to retrieve the results for a HIT.

        The returned ResultSet will have the following attributes:

        NumResults
                The number of assignments on the page in the filtered results
                list, equivalent to the number of assignments being returned
                by this call.
                A non-negative integer, as a string.
        PageNumber
                The number of the page in the filtered results list being
                returned.
                A positive integer, as a string.
        TotalNumResults
                The total number of HITs in the filtered results list based
                on this call.
                A non-negative integer, as a string.

        The ResultSet will contain zero or more Assignment objects

        """
    params = {'HITId': hit_id, 'SortProperty': sort_by, 'SortDirection': sort_direction, 'PageSize': page_size, 'PageNumber': page_number}
    if status is not None:
        params['AssignmentStatus'] = status
    if response_groups:
        self.build_list_params(params, response_groups, 'ResponseGroup')
    return self._process_request('GetAssignmentsForHIT', params, [('Assignment', Assignment)])