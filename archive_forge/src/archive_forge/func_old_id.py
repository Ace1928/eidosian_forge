def old_id(self, file_id):
    """Look up the original file id of a file.

        :param file_id: New file id
        :return: Old file id if mapped, otherwise new file id
        """
    for x in self.map:
        if self.map[x] == file_id:
            return x
    return file_id