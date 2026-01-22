import functools
import gettext
import logging
import os
import shutil
import sys
import warnings
import xml.dom.minidom
import xml.parsers.expat
import zipfile
def find_dictionaries(registry):

    def oor_name(name, element):
        return element.attributes['oor:name'].value.lower() == name

    def get_property(name, properties):
        property = list(filter(functools.partial(oor_name, name), properties))
        if property:
            return property[0].getElementsByTagName('value')[0]
    result = []
    for dictionaries in filter(functools.partial(oor_name, 'dictionaries'), registry.getElementsByTagName('node')):
        for dictionary in dictionaries.getElementsByTagName('node'):
            properties = dictionary.getElementsByTagName('prop')
            format = get_property('format', properties).firstChild.data.strip()
            if format and format == 'DICT_SPELL':
                locations = get_property('locations', properties)
                if locations.firstChild.nodeType == xml.dom.Node.TEXT_NODE:
                    locations = locations.firstChild.data
                    locations = locations.replace('%origin%/', '').strip()
                    result.append(locations.split())
                else:
                    locations = [item.firshChild.data.replace('%origin%/', '').strip() for item in locations.getElementsByTagName('it')]
                    result.append(locations)
    return result