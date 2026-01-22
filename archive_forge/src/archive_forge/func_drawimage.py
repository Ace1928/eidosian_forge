from ase.io.utils import PlottingVariables, make_patch_list
def drawimage(atoms):
    ax.clear()
    ax.axis('off')
    plot_atoms(atoms, ax=ax, **parameters)
    nframes[0] += 1
    if not hasattr(images, '__len__') and nframes[0] == save_count:
        import warnings
        warnings.warn('Number of frames reached animation savecount {}; some frames may not be saved.'.format(save_count))