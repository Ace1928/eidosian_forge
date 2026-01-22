import datetime
from lxml import etree
from oslo_log import log as logging
from oslo_serialization import jsonutils
def object_to_element(self, obj, element):
    if isinstance(obj, list):
        for item in obj:
            subelement = etree.SubElement(element, 'member')
            self.object_to_element(item, subelement)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            subelement = etree.SubElement(element, key)
            if key in JSON_ONLY_KEYS:
                if value:
                    try:
                        subelement.text = jsonutils.dumps(value)
                    except TypeError:
                        subelement.text = str(value)
            else:
                self.object_to_element(value, subelement)
    else:
        element.text = str(obj)