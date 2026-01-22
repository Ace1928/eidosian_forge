from copy import deepcopy
def rename_column_references(self, table, old_column, new_column):
    if self.table != table:
        return
    expressions = deepcopy(self.expressions)
    self.columns = []
    for col in self.compiler.query._gen_cols([expressions]):
        if col.target.column == old_column:
            col.target.column = new_column
        self.columns.append(col.target.column)
    self.expressions = expressions