from . import schema
from .jsonutil import get_column
from .search import Search
def look_for(self, element_name, datatype_name=None):
    paths = []
    self._intf.manage.schemas._init()
    if ':' in element_name:
        for root in self._intf.manage.schemas._trees.values():
            paths.extend(schema.datatype_attributes(root, element_name))
        return paths
    for xsd in self._intf.manage.schemas():
        trees = self._intf.manage.schemas._trees[xsd]
        if datatype_name is not None:
            datatypes = [datatype_name]
        else:
            datatypes = schema.datatypes(trees)
        for datatype in datatypes:
            for path in schema.datatype_attributes(trees, datatype):
                if element_name in path:
                    paths.append(path)
    return paths