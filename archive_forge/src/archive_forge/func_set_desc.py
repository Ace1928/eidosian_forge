from .trait_base import class_of
def set_desc(self, desc, object=None):
    if hasattr(self, 'desc'):
        if desc is not None:
            self.desc = desc
        if object is not None:
            self.object = object
        self.set_args()