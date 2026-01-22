def dump_range(self, code0, code1, set, file):
    if set:
        if code0 == -maxint:
            if code1 == maxint:
                k = 'any'
            else:
                k = '< %s' % self.dump_char(code1)
        elif code1 == maxint:
            k = '> %s' % self.dump_char(code0 - 1)
        elif code0 == code1 - 1:
            k = self.dump_char(code0)
        else:
            k = '%s..%s' % (self.dump_char(code0), self.dump_char(code1 - 1))
        self.dump_trans(k, set, file)