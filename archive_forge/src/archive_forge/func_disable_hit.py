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
def disable_hit(self, hit_id, response_groups=None):
    """
        Remove a HIT from the Mechanical Turk marketplace, approves all
        submitted assignments that have not already been approved or rejected,
        and disposes of the HIT and all assignment data.

        Assignments for the HIT that have already been submitted, but not yet
        approved or rejected, will be automatically approved. Assignments in
        progress at the time of the call to DisableHIT will be approved once
        the assignments are submitted. You will be charged for approval of
        these assignments.  DisableHIT completely disposes of the HIT and
        all submitted assignment data. Assignment results data cannot be
        retrieved for a HIT that has been disposed.

        It is not possible to re-enable a HIT once it has been disabled.
        To make the work from a disabled HIT available again, create a new HIT.
        """
    params = {'HITId': hit_id}
    if response_groups:
        self.build_list_params(params, response_groups, 'ResponseGroup')
    return self._process_request('DisableHIT', params)