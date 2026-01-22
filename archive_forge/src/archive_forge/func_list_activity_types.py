import time
import boto
from boto.connection import AWSAuthConnection
from boto.provider import Provider
from boto.exception import SWFResponseError
from boto.swf import exceptions as swf_exceptions
from boto.compat import json
def list_activity_types(self, domain, registration_status, name=None, maximum_page_size=None, next_page_token=None, reverse_order=None):
    """
        Returns information about all activities registered in the
        specified domain that match the specified name and
        registration status. The result includes information like
        creation date, current status of the activity, etc. The
        results may be split into multiple pages. To retrieve
        subsequent pages, make the call again using the nextPageToken
        returned by the initial call.

        :type domain: string
        :param domain: The name of the domain in which the activity
            types have been registered.

        :type registration_status: string
        :param registration_status: Specifies the registration status
            of the activity types to list.  Valid values are:

            * REGISTERED
            * DEPRECATED

        :type name: string
        :param name: If specified, only lists the activity types that
            have this name.

        :type maximum_page_size: integer
        :param maximum_page_size: The maximum number of results
            returned in each page. The default is 100, but the caller can
            override this value to a page size smaller than the
            default. You cannot specify a page size greater than 100.

        :type next_page_token: string
        :param next_page_token: If on a previous call to this method a
            NextResultToken was returned, the results have more than one
            page. To get the next page of results, repeat the call with
            the nextPageToken and keep all other arguments unchanged.

        :type reverse_order: boolean

        :param reverse_order: When set to true, returns the results in
            reverse order. By default the results are returned in
            ascending alphabetical order of the name of the activity
            types.

        :raises: SWFOperationNotPermittedError, UnknownResourceFault
        """
    return self.json_request('ListActivityTypes', {'domain': domain, 'name': name, 'registrationStatus': registration_status, 'maximumPageSize': maximum_page_size, 'nextPageToken': next_page_token, 'reverseOrder': reverse_order})