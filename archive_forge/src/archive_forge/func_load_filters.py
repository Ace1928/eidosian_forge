import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
def load_filters(filters_path):
    """Load filters from a list of directories."""
    filterlist = []
    for filterdir in filters_path:
        if not os.path.isdir(filterdir):
            continue
        for filterfile in filter(lambda f: not f.startswith('.'), os.listdir(filterdir)):
            filterfilepath = os.path.join(filterdir, filterfile)
            if not os.path.isfile(filterfilepath):
                continue
            kwargs = {'strict': False}
            filterconfig = configparser.RawConfigParser(**kwargs)
            filterconfig.read(filterfilepath)
            for name, value in filterconfig.items('Filters'):
                filterdefinition = [s.strip() for s in value.split(',')]
                newfilter = build_filter(*filterdefinition)
                if newfilter is None:
                    continue
                newfilter.name = name
                filterlist.append(newfilter)
    privsep = build_filter('CommandFilter', 'privsep-helper', 'root')
    privsep.name = 'privsep-helper'
    filterlist.append(privsep)
    return filterlist