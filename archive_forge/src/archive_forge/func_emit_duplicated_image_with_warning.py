import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
def emit_duplicated_image_with_warning(img, image_with):
    img_uuid_list = [str(image.id) for image in img]
    print(_('WARNING: Multiple matching images: %(img_uuid_list)s\nUsing image: %(chosen_one)s') % {'img_uuid_list': img_uuid_list, 'chosen_one': img_uuid_list[0]}, file=sys.stderr)