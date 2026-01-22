import pyparsing as pp
from pathlib import Path
def read_include_contents(s, l, t):
    include_file_ref = t.include_file_name
    include_echo = '/* {} */'.format(pp.line(l, s).strip())
    if include_file_ref not in seen:
        seen.add(include_file_ref)
        included_file_contents = Path(include_file_ref).read_text()
        return include_echo + '\n' + include_directive.transformString(included_file_contents)
    else:
        lead = ' ' * (pp.col(l, s) - 1)
        return '/* recursive include! */\n{}{}'.format(lead, include_echo)