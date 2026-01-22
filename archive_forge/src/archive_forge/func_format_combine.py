import re
import html
from paste.util import PySourceColor
def format_combine(self, data_by_importance, lines, exc_info):
    lines[:0] = [value for n, value in data_by_importance['important']]
    lines.append(exc_info)
    for name in ('normal', 'supplemental'):
        lines.extend([value for n, value in data_by_importance[name]])
    if data_by_importance['extra']:
        lines.append('<script type="text/javascript">\nshow_button(\'extra_data\', \'extra data\');\n</script>\n' + '<div id="extra_data" class="hidden-data">\n')
        lines.extend([value for n, value in data_by_importance['extra']])
        lines.append('</div>')
    text = self.format_combine_lines(lines)
    if self.include_reusable:
        return error_css + hide_display_js + text
    else:
        return text