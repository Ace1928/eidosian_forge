from struct import unpack
def read_enum(self):
    """An enum is encoded by a int, representing the zero-based position of the
        symbol in the schema.
        """
    return self.read_long()