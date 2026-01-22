from __future__ import annotations
import abc
import dataclasses
import datetime
import decimal
from xml.dom import minidom
from xml.etree import ElementTree as ET
def get_xml_element(self) -> ET.Element:
    """Return an XML element representing this instance."""
    element = ET.Element('testsuites', self.get_attributes())
    element.extend([suite.get_xml_element() for suite in self.suites])
    return element