import logging
def end_p(self):
    if self.do_p:
        self.doc.write('\n\n%s' % self.spaces())