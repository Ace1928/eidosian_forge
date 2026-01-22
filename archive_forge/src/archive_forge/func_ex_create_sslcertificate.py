import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_create_sslcertificate(self, name, certificate=None, private_key=None, description=None):
    """
        Creates a SslCertificate resource in the specified project using the
        data included in the request.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  name:  Name of the resource. Provided by the client when the
                       resource is created. The name must be 1-63 characters
                       long, and comply with RFC1035. Specifically, the name
                       must be 1-63 characters long and match the regular
                       expression [a-z]([-a-z0-9]*[a-z0-9])? which means the
                       first character must be a lowercase letter, and all
                       following characters must be a dash, lowercase letter,
                       or digit, except the last character, which cannot be a
                       dash.
        :type   name: ``str``

        :param  certificate:  A string containing local certificate file in
                              PEM format. The certificate chain
                              must be no greater than 5 certs long. The
                              chain must include at least one intermediate
                              cert.
        :type   certificate: ``str``

        :param  private_key:  A string containing a write-only private key
                              in PEM format. Only insert RPCs will include
                              this field.
        :type   private_key: ``str``

        :keyword  description:  An optional description of this resource.
                                Provide this property when you create the
                                resource.
        :type   description: ``str``

        :return:  `GCESslCertificate` object.
        :rtype: :class:`GCESslCertificate`
        """
    request = '/global/sslCertificates'
    request_data = {}
    request_data['name'] = name
    request_data['certificate'] = certificate
    request_data['privateKey'] = private_key
    request_data['description'] = description
    self.connection.async_request(request, method='POST', data=request_data)
    return self.ex_get_sslcertificate(name)