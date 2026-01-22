import logging
def link_target_definition(self, refname, link):
    self.doc.writeln(f'.. _{refname}: {link}')