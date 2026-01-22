from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient
def list_csi_driver(self, **kwargs):
    """
        list or watch objects of kind CSIDriver
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.list_csi_driver(async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str pretty: If 'true', then the output is pretty printed.
        :param str _continue: The continue option should be set when retrieving
        more results from the server. Since this value is server defined,
        clients may only use the continue value from a previous query result
        with identical query parameters (except for the value of continue) and
        the server may reject a continue value it does not recognize. If the
        specified continue value is no longer valid whether due to expiration
        (generally five to fifteen minutes) or a configuration change on the
        server, the server will respond with a 410 ResourceExpired error
        together with a continue token. If the client needs a consistent list,
        it must restart their list without the continue field. Otherwise, the
        client may send another list request with the token received with the
        410 error, the server will respond with a list starting from the next
        key, but from the latest snapshot, which is inconsistent from the
        previous list results - objects that are created, modified, or deleted
        after the first list request will be included in the response, as long
        as their keys are after the "next key".  This field is not supported
        when watch is true. Clients may start a watch from the last
        resourceVersion value returned by the server and not miss any
        modifications.
        :param str field_selector: A selector to restrict the list of returned
        objects by their fields. Defaults to everything.
        :param str label_selector: A selector to restrict the list of returned
        objects by their labels. Defaults to everything.
        :param int limit: limit is a maximum number of responses to return for a
        list call. If more items exist, the server will set the `continue` field
        on the list metadata to a value that can be used with the same initial
        query to retrieve the next set of results. Setting a limit may return
        fewer than the requested amount of items (up to zero items) in the event
        all requested objects are filtered out and clients should only use the
        presence of the continue field to determine whether more results are
        available. Servers may choose not to support the limit argument and will
        return all of the available results. If limit is specified and the
        continue field is empty, clients may assume that no more results are
        available. This field is not supported if watch is true.  The server
        guarantees that the objects returned when using continue will be
        identical to issuing a single list call without a limit - that is, no
        objects created, modified, or deleted after the first request is issued
        will be included in any subsequent continued requests. This is sometimes
        referred to as a consistent snapshot, and ensures that a client that is
        using limit to receive smaller chunks of a very large result can ensure
        they see all possible objects. If objects are updated during a chunked
        list the version of the object that was present at the time the first
        list result was calculated is returned.
        :param str resource_version: When specified with a watch call, shows
        changes that occur after that particular version of a resource. Defaults
        to changes from the beginning of history. When specified for list: - if
        unset, then the result is returned from remote storage based on
        quorum-read flag; - if it's 0, then we simply return what we currently
        have in cache, no guarantee; - if set to non zero, then the result is at
        least as fresh as given rv.
        :param int timeout_seconds: Timeout for the list/watch call. This limits
        the duration of the call, regardless of any activity or inactivity.
        :param bool watch: Watch for changes to the described resources and
        return them as a stream of add, update, and remove notifications.
        Specify resourceVersion.
        :return: V1beta1CSIDriverList
                 If the method is called asynchronously,
                 returns the request thread.
        """
    kwargs['_return_http_data_only'] = True
    if kwargs.get('async_req'):
        return self.list_csi_driver_with_http_info(**kwargs)
    else:
        data = self.list_csi_driver_with_http_info(**kwargs)
        return data