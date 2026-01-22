import typing as t
def jsonValue(self) -> t.Dict[str, t.Any]:
    return {'type': self.typeName(), 'fields': [x.jsonValue() for x in self]}