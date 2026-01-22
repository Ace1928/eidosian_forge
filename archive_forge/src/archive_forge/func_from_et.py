from os_ken.lib import stringify
from lxml import objectify
import lxml.etree as ET
@classmethod
def from_et(cls, et):
    return cls(raw_et=et)