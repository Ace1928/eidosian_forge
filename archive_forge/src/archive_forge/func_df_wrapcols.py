import mplfinance as mpf
import pandas as pd
import textwrap
def df_wrapcols(df, wrap_columns=None):
    if wrap_columns is None:
        return df
    if not isinstance(wrap_columns, dict):
        raise TypeError('wrap_columns must be a dict of column_names and wrap_lengths')
    for col in wrap_columns:
        if col not in df.columns:
            raise ValueError('column "' + str(col) + '" not found in df.columns')
    index = []
    column_data = {}
    for col in df.columns:
        column_data[col] = []
    for ix in df.index:
        row = df.loc[ix,]
        row_data = {}
        for col in row.index:
            cstr = str(row[col])
            if col in wrap_columns:
                wlen = wrap_columns[col]
                tw = textwrap.wrap(cstr, wlen) if not cstr.isspace() else [' ']
            else:
                tw = [cstr]
            row_data[col] = tw
        cmax = max(row_data, key=lambda k: len(row_data[k]))
        rlen = len(row_data[cmax])
        for r in range(rlen):
            for col in row.index:
                extension = [' '] * (rlen - len(row_data[col]))
                row_data[col].extend(extension)
                column_data[col].append(row_data[col][r])
            ixstr = str(ix) + '.' + str(r) if r > 0 else str(ix)
            index.append(ixstr)
    return pd.DataFrame(column_data, index=index)