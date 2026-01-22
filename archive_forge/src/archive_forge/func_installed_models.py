from collections import namedtuple
def installed_models(self, tables):
    """
        Return a set of all models represented by the provided list of table
        names.
        """
    tables = set(map(self.identifier_converter, tables))
    return {m for m in self.get_migratable_models() if self.identifier_converter(m._meta.db_table) in tables}