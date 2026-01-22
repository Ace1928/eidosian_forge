from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def list_presets(self, ascending=None, page_token=None):
    """
        The ListPresets operation gets a list of the default presets
        included with Elastic Transcoder and the presets that you've
        added in an AWS region.

        :type ascending: string
        :param ascending: To list presets in chronological order by the date
            and time that they were created, enter `True`. To list presets in
            reverse chronological order, enter `False`.

        :type page_token: string
        :param page_token: When Elastic Transcoder returns more than one page
            of results, use `pageToken` in subsequent `GET` requests to get
            each successive page of results.

        """
    uri = '/2012-09-25/presets'.format()
    params = {}
    if ascending is not None:
        params['Ascending'] = ascending
    if page_token is not None:
        params['PageToken'] = page_token
    return self.make_request('GET', uri, expected_status=200, params=params)