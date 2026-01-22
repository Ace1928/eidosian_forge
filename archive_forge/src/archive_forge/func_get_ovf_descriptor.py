from __future__ import absolute_import, division, print_function
import hashlib
import io
import os
import re
import ssl
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
from threading import Thread
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.request import Request, urlopen
from ansible.module_utils.urls import generic_urlparse, open_url, urlparse, urlunparse
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def get_ovf_descriptor(self):
    if self.params['url'] is None:
        try:
            path_exists(self.params['ovf'])
        except ValueError as e:
            self.module.fail_json(msg='%s' % e)
        if tarfile.is_tarfile(self.params['ovf']):
            self.tar = tarfile.open(self.params['ovf'])
            ovf = None
            for candidate in self.tar.getmembers():
                dummy, ext = os.path.splitext(candidate.name)
                if ext.lower() == '.ovf':
                    ovf = candidate
                    break
            if not ovf:
                self.module.fail_json(msg='Could not locate OVF file in %(ovf)s' % self.params)
            self.ovf_descriptor = to_native(self.tar.extractfile(ovf).read())
        else:
            with open(self.params['ovf'], encoding='utf-8') as f:
                self.ovf_descriptor = f.read()
        return self.ovf_descriptor
    else:
        self.handle = WebHandle(self.params['url'])
        self.tar = tarfile.open(fileobj=self.handle)
        ovffilename = list(filter(lambda x: x.endswith('.ovf'), self.tar.getnames()))[0]
        ovffile = self.tar.extractfile(ovffilename)
        self.ovf_descriptor = ovffile.read().decode()
        if self.ovf_descriptor:
            return self.ovf_descriptor
        else:
            self.module.fail_json(msg='Could not locate OVF file in %(url)s' % self.params)