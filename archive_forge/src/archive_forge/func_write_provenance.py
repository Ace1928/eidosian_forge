from collections import OrderedDict
from copy import deepcopy
from pickle import dumps
import os
import getpass
import platform
from uuid import uuid1
import simplejson as json
import numpy as np
import prov.model as pm
from .. import get_info, logging, __version__
from .filemanip import md5, hashlib, hash_infile
def write_provenance(self, filename='provenance', format='all'):
    if format in ['provn', 'all']:
        with open(filename + '.provn', 'wt') as fp:
            fp.writelines(self.g.get_provn())
    try:
        if format in ['rdf', 'all']:
            if len(self.g.bundles) == 0:
                rdf_format = 'turtle'
                ext = '.ttl'
            else:
                rdf_format = 'trig'
                ext = '.trig'
            self.g.serialize(filename + ext, format='rdf', rdf_format=rdf_format)
        if format in ['jsonld']:
            self.g.serialize(filename + '.jsonld', format='rdf', rdf_format='json-ld', indent=4)
    except pm.serializers.DoNotExist:
        pass
    return self.g