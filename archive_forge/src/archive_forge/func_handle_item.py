from xml.parsers import expat
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
def handle_item(path, item):
    marshal.dump((path, item), sys.stdout)
    return True