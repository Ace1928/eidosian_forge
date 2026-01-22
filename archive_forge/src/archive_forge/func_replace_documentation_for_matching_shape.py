import inspect
import jmespath
from botocore.compat import six
def replace_documentation_for_matching_shape(self, event_name, section, **kwargs):
    if self._shape_name == section.context.get('shape'):
        self._replace_documentation(event_name, section)
    for section_name in section.available_sections:
        sub_section = section.get_section(section_name)
        if self._shape_name == sub_section.context.get('shape'):
            self._replace_documentation(event_name, sub_section)
        else:
            self.replace_documentation_for_matching_shape(event_name, sub_section)