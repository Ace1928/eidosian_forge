import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def document_output(self, section, example, shape):
    output_section = section.add_new_section('output')
    output_section.style.new_line()
    output_section.write('Expected Output:')
    output_section.style.new_line()
    output_section.style.start_codeblock()
    params = example.get('output', {})
    params['ResponseMetadata'] = {'...': '...'}
    comments = example.get('comments')
    if comments:
        comments = comments.get('output')
    self._document_dict(output_section, params, comments, [], shape, True)
    closing_section = output_section.add_new_section('output-close')
    closing_section.style.end_codeblock()