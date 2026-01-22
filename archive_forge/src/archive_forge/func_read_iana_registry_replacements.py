import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def read_iana_registry_replacements():
    replacements = {}
    for entry in parse_registry():
        if entry['Type'] == 'language' and 'Preferred-Value' in entry:
            replacements[entry['Subtag']] = entry['Preferred-Value']
        elif 'Tag' in entry and 'Preferred-Value' in entry:
            replacements[entry['Tag'].lower()] = entry['Preferred-Value']
    return replacements