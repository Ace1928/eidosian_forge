import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def match_data(self, data, max_pri=100, min_pri=0, possible=None):
    """Do magic sniffing on some bytes.
        
        max_pri & min_pri can be used to specify the maximum & minimum priority
        rules to look for. possible can be a list of mimetypes to check, or None
        (the default) to check all mimetypes until one matches.
        
        Returns the MIMEtype found, or None if no entries match.
        """
    if possible is not None:
        types = []
        for mt in possible:
            for pri, rule in self.bytype[mt]:
                types.append((pri, mt, rule))
        types.sort(key=lambda x: x[0])
    else:
        types = self.alltypes
    for priority, mimetype, rule in types:
        if priority > max_pri:
            continue
        if priority < min_pri:
            break
        if rule.match(data):
            return mimetype