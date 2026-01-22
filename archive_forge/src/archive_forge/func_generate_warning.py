from ..host        import HostLanguage
def generate_warning(node, attr):
    if attr == 'rel':
        msg = 'Attribute @rel should not be used in RDFa Lite (consider using @property)'
    elif attr == 'about':
        msg = 'Attribute @about should not be used in RDFa Lite (consider using a <link> element with @href or @resource)'
    else:
        msg = 'Attribute @%s should not be used in RDFa Lite' % attr
    options.add_warning(msg, node=node)