def handle_role(node):
    if node.hasAttribute('role'):
        old_values = node.getAttribute('role').strip().split()
        new_values = ''
        for val in old_values:
            if termname.match(val):
                new_values += XHTML_URI + val + ' '
            else:
                new_values += val + ' '
        node.setAttribute('role', new_values.strip())