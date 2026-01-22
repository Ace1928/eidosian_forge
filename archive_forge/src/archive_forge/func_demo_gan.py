import ray
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def demo_gan(checkpoint_paths):
    img_list = []
    fixed_noise = torch.randn(64, nz, 1, 1)
    for path in checkpoint_paths:
        checkpoint_dict = torch.load(os.path.join(path, 'checkpoint.pt'))
        loadedG = Generator()
        loadedG.load_state_dict(checkpoint_dict['netGmodel'])
        with torch.no_grad():
            fake = loadedG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save('./generated.gif', writer='imagemagick', dpi=72)
    plt.show()