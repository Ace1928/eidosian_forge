from __future__ import absolute_import, division, print_function
import json
import logging
import optparse
import os
import ssl
import sys
import time
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.module_utils.six import integer_types, text_type, string_types
from ansible.module_utils.six.moves import configparser
from psphere.client import Client
from psphere.errors import ObjectNotFoundError
from psphere.managedobjects import HostSystem, VirtualMachine, ManagedObject, ClusterComputeResource
from suds.sudsobject import Object as SudsObject

        Read info about a specific host or VM from cache or VMware API.
        