import io
import logging
import os
import sys
import urllib
from osc_lib import utils
from openstackclient.api import api
def object_list(self, container=None, full_listing=False, limit=None, marker=None, end_marker=None, delimiter=None, prefix=None, **params):
    """List objects in a container

        :param string container:
            container name to get a listing for
        :param boolean full_listing:
            if True, return a full listing, else returns a max of
            10000 listings
        :param integer limit:
            query return count limit
        :param string marker:
            query marker
        :param string end_marker:
            query end_marker
        :param string prefix:
            query prefix
        :param string delimiter:
            string to delimit the queries on
        :returns: a tuple of (response headers, a list of objects) The response
            headers will be a dict and all header names will be lowercase.
        """
    if container is None:
        return None
    params['format'] = 'json'
    if full_listing:
        data = listing = self.object_list(container=container, limit=limit, marker=marker, end_marker=end_marker, prefix=prefix, delimiter=delimiter, **params)
        while listing:
            if delimiter:
                marker = listing[-1].get('name', listing[-1].get('subdir'))
            else:
                marker = listing[-1]['name']
            listing = self.object_list(container=container, limit=limit, marker=marker, end_marker=end_marker, prefix=prefix, delimiter=delimiter, **params)
            if listing:
                data.extend(listing)
        return data
    if limit:
        params['limit'] = limit
    if marker:
        params['marker'] = marker
    if end_marker:
        params['end_marker'] = end_marker
    if prefix:
        params['prefix'] = prefix
    if delimiter:
        params['delimiter'] = delimiter
    return self.list(urllib.parse.quote(container), **params)