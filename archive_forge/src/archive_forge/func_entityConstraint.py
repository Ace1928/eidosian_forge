from twisted.python import reflect
def entityConstraint(self, entity):
    if isinstance(entity, self.entityType):
        return 1
    else:
        raise ConstraintViolation(f'{entity} of incorrect type ({self.entityType})')