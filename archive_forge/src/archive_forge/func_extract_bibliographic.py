import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def extract_bibliographic(self, field_list):
    docinfo = nodes.docinfo()
    bibliofields = self.language.bibliographic_fields
    labels = self.language.labels
    topics = {'dedication': None, 'abstract': None}
    for field in field_list:
        try:
            name = field[0][0].astext()
            normedname = nodes.fully_normalize_name(name)
            if not (len(field) == 2 and normedname in bibliofields and self.check_empty_biblio_field(field, name)):
                raise TransformError
            canonical = bibliofields[normedname]
            biblioclass = self.biblio_nodes[canonical]
            if issubclass(biblioclass, nodes.TextElement):
                if not self.check_compound_biblio_field(field, name):
                    raise TransformError
                utils.clean_rcs_keywords(field[1][0], self.rcs_keyword_substitutions)
                docinfo.append(biblioclass('', '', *field[1][0]))
            elif issubclass(biblioclass, nodes.authors):
                self.extract_authors(field, name, docinfo)
            elif issubclass(biblioclass, nodes.topic):
                if topics[canonical]:
                    field[-1] += self.document.reporter.warning('There can only be one "%s" field.' % name, base_node=field)
                    raise TransformError
                title = nodes.title(name, labels[canonical])
                title[0].rawsource = labels[canonical]
                topics[canonical] = biblioclass('', title, *field[1].children, classes=[canonical])
            else:
                docinfo.append(biblioclass('', *field[1].children))
        except TransformError:
            if len(field[-1]) == 1 and isinstance(field[-1][0], nodes.paragraph):
                utils.clean_rcs_keywords(field[-1][0], self.rcs_keyword_substitutions)
            classvalue = nodes.make_id(normedname)
            if classvalue:
                field['classes'].append(classvalue)
            docinfo.append(field)
    nodelist = []
    if len(docinfo) != 0:
        nodelist.append(docinfo)
    for name in ('dedication', 'abstract'):
        if topics[name]:
            nodelist.append(topics[name])
    return nodelist