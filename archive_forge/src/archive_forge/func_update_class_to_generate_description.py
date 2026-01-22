def update_class_to_generate_description(class_to_generate):
    import textwrap
    description = class_to_generate['description']
    lines = []
    for line in description.splitlines():
        wrapped = textwrap.wrap(line.strip(), 100)
        lines.extend(wrapped)
        lines.append('')
    while lines and lines[-1] == '':
        lines = lines[:-1]
    class_to_generate['description'] = '    ' + '\n    '.join(lines)