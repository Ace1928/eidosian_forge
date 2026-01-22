from ..pari import pari
import fractions
def downright(row):
    return [1 if row == col else 0 for col in range(num_cols_rows)]