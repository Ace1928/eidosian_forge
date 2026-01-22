def new_id(self, file_id):
    """Look up the new file id of a file.

        :param file_id: Old file id
        :return: New file id
        """
    try:
        return self.map[file_id]
    except KeyError:
        return file_id