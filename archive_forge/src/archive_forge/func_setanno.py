import enum
import gast
def setanno(node, key, value, field_name='___pyct_anno'):
    annotations = getattr(node, field_name, {})
    setattr(node, field_name, annotations)
    annotations[key] = value
    if field_name not in node._fields:
        node._fields += (field_name,)