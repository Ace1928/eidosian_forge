from inspect import isclass
def sgenerate(self):
    return self.generate(lambda obj: obj.sgenerate())