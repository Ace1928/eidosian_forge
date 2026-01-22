@property
def legacy_ro(self):
    if self.__legacy_ro is None:
        self.__legacy_ro = tuple(_legacy_ro(self.leaf))
    return list(self.__legacy_ro)