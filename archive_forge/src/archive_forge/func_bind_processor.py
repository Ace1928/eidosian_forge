from ... import types as sqltypes
def bind_processor(self, dialect):
    super_proc = self.string_bind_processor(dialect)

    def process(value):
        value = self._format_value(value)
        if super_proc:
            value = super_proc(value)
        return value
    return process