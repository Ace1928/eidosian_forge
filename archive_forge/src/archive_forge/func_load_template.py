import six
from genshi.input import ET, HTML, XML
from genshi.output import DocType
from genshi.template.base import Template
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
from genshi.template.text import TextTemplate, NewTextTemplate
def load_template(self, templatename, template_string=None):
    """Find a template specified in python 'dot' notation, or load one from
        a string.
        """
    if template_string is not None:
        return self.template_class(template_string)
    if self.use_package_naming:
        divider = templatename.rfind('.')
        if divider >= 0:
            from pkg_resources import resource_filename
            package = templatename[:divider]
            basename = templatename[divider + 1:] + self.extension
            templatename = resource_filename(package, basename)
    return self.loader.load(templatename)