import re
def query_coverage(self):
    """Return the length of the query covered in the alignment."""
    s = self.query_aln.replace('=', '')
    return len(s)