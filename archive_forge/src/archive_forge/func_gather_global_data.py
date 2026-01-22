from collections import OrderedDict
import importlib
def gather_global_data(self, local_data):
    assert len(local_data) == len(self._local_map)
    if not self._mpi_interface.have_mpi:
        return list(local_data)
    comm = self._mpi_interface.comm
    global_data_list_of_lists = comm.gather(local_data)
    if global_data_list_of_lists is not None:
        return self._stack_global_data(global_data_list_of_lists)
    assert self.is_root() == False
    return None