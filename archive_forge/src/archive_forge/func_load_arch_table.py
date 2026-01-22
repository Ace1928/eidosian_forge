import os
import collections.abc
@classmethod
def load_arch_table(cls, path='/usr/share/dpkg'):
    """Load the Dpkg Architecture Table

        This class method loads the architecture table from dpkg, so it can be used.

        >>> arch_table = DpkgArchTable.load_arch_table()
        >>> arch_table.matches_architecture("amd64", "any")
        True

        The method assumes the dpkg "tuple arch" format version 1.0 or the older triplet format.

        :param path: Choose a different directory for loading the architecture data.  The provided
          directory must contain the architecture data files from dpkg (such as "tupletable" and
          "cputable")
        """
    tupletable_path = os.path.join(path, 'tupletable')
    cputable_path = os.path.join(path, 'cputable')
    triplet_compat = False
    if not os.path.exists(tupletable_path):
        triplettable_path = os.path.join(path, 'triplettable')
        if os.path.join(triplettable_path):
            triplet_compat = True
            tupletable_path = triplettable_path
    with open(tupletable_path, encoding='utf-8') as tuple_fd, open(cputable_path, encoding='utf-8') as cpu_fd:
        return cls._from_file(tuple_fd, cpu_fd, triplet_compat=triplet_compat)