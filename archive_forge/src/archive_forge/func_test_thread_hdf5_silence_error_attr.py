import threading
import h5py
def test_thread_hdf5_silence_error_attr(tmp_path, capfd):
    """Verify the error printing is squashed in all threads.

    No console messages should be shown for non-existing attributes
    """

    def test():
        with h5py.File(tmp_path / 'test.h5', 'w') as newfile:
            newfile['newdata'] = [1, 2, 3]
            try:
                nonexistent_attr = newfile['newdata'].attrs['nonexistent_attr']
            except KeyError:
                pass
    th = threading.Thread(target=test)
    th.start()
    th.join()
    captured = capfd.readouterr()
    assert captured.err == ''
    assert captured.out == ''