def make_literal(value):
    value = html.escape(value, 1)
    value = value.replace('\n\r', '\n')
    value = value.replace('\r', '\n')
    value = value.replace('\n', '<br>\n')
    return value