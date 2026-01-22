import logging
def external_link(self, title, link):
    if self.doc.target == 'html':
        self.doc.write(f'`{title} <{link}>`_')
    else:
        self.doc.write(title)