import sys, re
def repr_attribute(self, attrs, name):
    if name == 'class_':
        value = getattr(attrs, name)
        if value is None:
            return
    return super(HtmlVisitor, self).repr_attribute(attrs, name)