from sys import stderr
def once(self, warning):
    if warning not in self.uttered:
        if self.enabled:
            logger.write(self.pfx + warning)
        self.uttered[warning] = 1