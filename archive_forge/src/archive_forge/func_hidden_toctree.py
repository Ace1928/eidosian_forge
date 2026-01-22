import logging
def hidden_toctree(self):
    if self.doc.target == 'html':
        self.doc.write('\n.. toctree::\n')
        self.doc.write('  :maxdepth: 1\n')
        self.doc.write('  :hidden:\n\n')