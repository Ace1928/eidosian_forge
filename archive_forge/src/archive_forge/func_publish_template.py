import hashlib
import json
import random
from urllib import parse
from swiftclient import utils as swiftclient_utils
import yaml
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def publish_template(self, contents, cleanup=True):
    oc = self.object_client
    oc.put_object(self.object_container_name, 'template.yaml', contents)
    if cleanup:
        self.addCleanup(oc.delete_object, self.object_container_name, 'template.yaml')
    path = '/v1/AUTH_%s/%s/%s' % (self.project_id, self.object_container_name, 'template.yaml')
    timeout = self.conf.build_timeout * 10
    tempurl = swiftclient_utils.generate_temp_url(path, timeout, self.swift_key, 'GET')
    sw_url = parse.urlparse(oc.url)
    return '%s://%s%s' % (sw_url.scheme, sw_url.netloc, tempurl)