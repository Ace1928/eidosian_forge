import copy
import importlib
import logging
import re
from warnings import warn as _warn
from saml2 import saml
from saml2 import xmlenc
from saml2.attribute_converter import ac_factory
from saml2.attribute_converter import from_local
from saml2.attribute_converter import get_local_name
from saml2.s_utils import MissingValue
from saml2.s_utils import assertion_factory
from saml2.s_utils import factory
from saml2.s_utils import sid
from saml2.saml import NAME_FORMAT_URI
from saml2.time_util import in_a_while
from saml2.time_util import instant
def post_entity_categories(maps, sp_entity_id=None, mds=None, required=None):
    restrictions = {}
    required_friendly_names = [d.get('friendly_name') or get_local_name(acs=self.acs, attr=d['name'], name_format=d['name_format']) for d in required or []]
    required = [friendly_name.lower() for friendly_name in required_friendly_names]
    if mds:
        ecs = mds.entity_categories(sp_entity_id)
        for ec_map in maps:
            for key, (atlist, only_required, no_aggregation) in ec_map.items():
                if key == '':
                    attrs = atlist
                elif isinstance(key, tuple):
                    if only_required:
                        attrs = [a for a in atlist if a in required]
                    else:
                        attrs = atlist
                    for _key in key:
                        if _key not in ecs:
                            attrs = []
                            break
                elif key in ecs:
                    if only_required:
                        attrs = [a for a in atlist if a in required]
                    else:
                        attrs = atlist
                else:
                    attrs = []
                if attrs and no_aggregation:
                    restrictions = {}
                for attr in attrs:
                    restrictions[attr] = None
                else:
                    restrictions[''] = None
    return restrictions