from xml.parsers import expat
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
def push_data(self, item, key, data):
    if self.postprocessor is not None:
        result = self.postprocessor(self.path, key, data)
        if result is None:
            return item
        key, data = result
    if item is None:
        item = self.dict_constructor()
    try:
        value = item[key]
        if isinstance(value, list):
            value.append(data)
        else:
            item[key] = [value, data]
    except KeyError:
        if key in self.force_list:
            item[key] = [data]
        else:
            item[key] = data
    return item