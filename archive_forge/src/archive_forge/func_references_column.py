from copy import deepcopy
def references_column(self, table, column):
    return any((hasattr(part, 'references_column') and part.references_column(table, column) for part in self.parts.values()))