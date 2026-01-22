import argparse
import datetime
import os
import sys
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_log import log
from oslo_serialization import jsonutils
import pbr.version
from keystone.cmd import bootstrap
from keystone.cmd import doctor
from keystone.cmd import idutils
from keystone.common import driver_hints
from keystone.common import fernet_utils
from keystone.common import jwt_utils
from keystone.common import sql
from keystone.common.sql import upgrades
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.federation import idp
from keystone.federation import utils as mapping_engine
from keystone.i18n import _
from keystone.server import backends
@staticmethod
def get_user_group():
    keystone_user_id = None
    keystone_group_id = None
    try:
        a = CONF.command.keystone_user
        if a:
            keystone_user_id = utils.get_unix_user(a)[0]
    except KeyError:
        raise ValueError("Unknown user '%s' in --keystone-user" % a)
    try:
        a = CONF.command.keystone_group
        if a:
            keystone_group_id = utils.get_unix_group(a)[0]
    except KeyError:
        raise ValueError("Unknown group '%s' in --keystone-group" % a)
    return (keystone_user_id, keystone_group_id)