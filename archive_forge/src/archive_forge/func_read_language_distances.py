import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def read_language_distances():
    language_info_path = data_filename('cldr/common/supplemental/languageInfo.xml')
    root = ET.fromstring(open(language_info_path).read())
    matches = root.findall('./languageMatching/languageMatches[@type="written_new"]/languageMatch')
    tag_distances = {}
    for match in matches:
        attribs = match.attrib
        n_parts = attribs['desired'].count('_') + 1
        if n_parts < 3:
            if attribs.get('oneway') == 'true':
                pairs = [(attribs['desired'], attribs['supported'])]
            else:
                pairs = [(attribs['desired'], attribs['supported']), (attribs['supported'], attribs['desired'])]
            for desired, supported in pairs:
                desired_distance = tag_distances.setdefault(desired, {})
                desired_distance[supported] = int(attribs['distance'])
                if desired == 'sh' or supported == 'sh':
                    if desired == 'sh':
                        desired = 'sr'
                    if supported == 'sh':
                        supported = 'sr'
                    if desired != supported:
                        desired_distance = tag_distances.setdefault(desired, {})
                        desired_distance[supported] = int(attribs['distance']) + 1
    return tag_distances