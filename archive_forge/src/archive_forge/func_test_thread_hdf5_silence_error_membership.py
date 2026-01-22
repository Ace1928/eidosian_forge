import threading
import h5py
def test_thread_hdf5_silence_error_membership(tmp_path, capfd):
    """Verify the error printing is squashed in all threads.

    No console messages should be shown from membership tests
    """
    th = threading.Thread(target=_access_not_existing_object, args=(tmp_path / 'test.h5',))
    th.start()
    th.join()
    captured = capfd.readouterr()
    assert captured.err == ''
    assert captured.out == ''