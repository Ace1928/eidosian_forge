import numpy as np
Return string for ReST table with entries `cell_values`

    Parameters
    ----------
    cell_values : (R, C) array-like
        At least 2D.  Can be greater than 2D, in which case you should adapt
        the `val_fmt` to deal with the multiple entries that will go in each
        cell
    row_names : None or (R,) length sequence, optional
        Row names.  If None, use ``row[0]`` etc.
    col_names : None or (C,) length sequence, optional
        Column names.  If None, use ``col[0]`` etc.
    title : str, optional
        Title for table.  Add as heading above table
    val_fmt : str, optional
        Format string using string ``format`` method mini-language. Converts
        the result of ``cell_values[r, c]`` to a string to make the cell
        contents. Default assumes a floating point value in a 2D `cell_values`.
    format_chars : None or dict, optional
        With keys 'down', 'along', 'thick_long', 'cross' and 'title_heading'.
        Values are characters for: lines going down; lines going along; thick
        lines along; two lines crossing; and the title overline / underline.
        All missing values filled with rst defaults.

    Returns
    -------
    table_str : str
        Multiline string with ascii table, suitable for printing
    