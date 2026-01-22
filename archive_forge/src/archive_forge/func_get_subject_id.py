import re
from lxml import etree
from .jsonutil import JsonTable
def get_subject_id(location):
    f = open(location, 'rb')
    content = f.read()
    f.close()
    subject = re.findall('<xnat:Subject\\sID="(.*?)"\\s', content)
    if subject != []:
        return subject[0]