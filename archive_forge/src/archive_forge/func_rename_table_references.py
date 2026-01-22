from copy import deepcopy
def rename_table_references(self, old_table, new_table):
    if self.table != old_table:
        return
    self.expressions = self.expressions.relabeled_clone({old_table: new_table})
    super().rename_table_references(old_table, new_table)