def read_tfm_file(file_name):
    with open(file_name, 'rb') as f:
        reader = TfmReader(f)
        reader.read_halfword()
        header_size = reader.read_halfword()
        start_char = reader.read_halfword()
        end_char = reader.read_halfword()
        width_table_size = reader.read_halfword()
        height_table_size = reader.read_halfword()
        depth_table_size = reader.read_halfword()
        italic_table_size = reader.read_halfword()
        ligkern_table_size = reader.read_halfword()
        kern_table_size = reader.read_halfword()
        reader.read_halfword()
        reader.read_halfword()
        reader.read_word()
        reader.read_fixword()
        if header_size > 2:
            reader.read_bcpl(40)
        if header_size > 12:
            reader.read_bcpl(20)
        for i in range(header_size - 17):
            reader.read_word()
        char_info = []
        for i in range(start_char, end_char + 1):
            char_info.append(CharInfoWord(reader.read_word()))
        width_table = []
        for i in range(width_table_size):
            width_table.append(reader.read_fixword())
        height_table = []
        for i in range(height_table_size):
            height_table.append(reader.read_fixword())
        depth_table = []
        for i in range(depth_table_size):
            depth_table.append(reader.read_fixword())
        italic_table = []
        for i in range(italic_table_size):
            italic_table.append(reader.read_fixword())
        ligkern_table = []
        for i in range(ligkern_table_size):
            skip = reader.read_byte()
            next_char = reader.read_byte()
            op = reader.read_byte()
            remainder = reader.read_byte()
            ligkern_table.append((skip, next_char, op, remainder))
        kern_table = []
        for i in range(kern_table_size):
            kern_table.append(reader.read_fixword())
        return TfmFile(start_char, end_char, char_info, width_table, height_table, depth_table, italic_table, ligkern_table, kern_table)