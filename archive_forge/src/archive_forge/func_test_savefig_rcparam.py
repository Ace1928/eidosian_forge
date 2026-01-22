import os
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
@pytest.mark.backend('macosx')
def test_savefig_rcparam(monkeypatch, tmp_path):

    def new_choose_save_file(title, directory, filename):
        assert directory == str(tmp_path)
        os.makedirs(f'{directory}/test')
        return f'{directory}/test/{filename}'
    monkeypatch.setattr(_macosx, 'choose_save_file', new_choose_save_file)
    fig = plt.figure()
    with mpl.rc_context({'savefig.directory': tmp_path}):
        fig.canvas.toolbar.save_figure()
        save_file = f'{tmp_path}/test/{fig.canvas.get_default_filename()}'
        assert os.path.exists(save_file)
        assert mpl.rcParams['savefig.directory'] == f'{tmp_path}/test'