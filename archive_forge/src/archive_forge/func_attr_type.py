import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def attr_type(self, name, scope, value):
    """Infer the attribute type of data named name. Currently this only
        supports inference of numeric types.

        If self.infer_numeric_types is false, type is used. Otherwise, pick the
        most general of types found across all values with name and scope. This
        means edges with data named 'weight' are treated separately from nodes
        with data named 'weight'.
        """
    if self.infer_numeric_types:
        types = self.attribute_types[name, scope]
        if len(types) > 1:
            types = {self.get_xml_type(t) for t in types}
            if 'string' in types:
                return str
            elif 'float' in types or 'double' in types:
                return float
            else:
                return int
        else:
            return list(types)[0]
    else:
        return type(value)