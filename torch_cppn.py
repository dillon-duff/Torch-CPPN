import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import uuid

import torch
import torch.nn as nn
import random

act_funcs = [
    nn.ELU(),
    nn.Hardshrink(),
    nn.Hardsigmoid(),
    nn.Hardtanh(),
    nn.Hardswish(),
    nn.LeakyReLU(),
    nn.LogSigmoid(),
    nn.PReLU(),
    nn.ReLU(),
    nn.ReLU6(),
    nn.RReLU(),
    nn.SELU(),
    nn.CELU(),
    nn.GELU(),
    nn.Sigmoid(),
    nn.SiLU(),
    nn.Identity(),
    nn.Mish(),
    nn.Softplus(),
    nn.Softshrink(),
    nn.Softsign(),
    nn.Tanh(),
    nn.Tanhshrink(),
]


class RandomNetwork(nn.Module):
    def __init__(self):
        super(RandomNetwork, self).__init__()

        num_layers = random.randint(2, 20)

        layer_sizes = [random.randint(2, 100) for _ in range(num_layers)]

        input_size = 3

        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(random.choice(act_funcs))
            input_size = size

        layers.append(nn.Linear(layer_sizes[-1], 1))
        layers.append(random.choice(act_funcs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


cool_cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
              'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
              'tab20c'] + ['twilight', 'twilight_shifted', 'hsv'] + ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                                                                     'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'] + ['viridis', 'plasma', 'inferno', 'magma', 'cividis']


def dist_from_origin(pos, origin=(0, 0)):
    return np.sqrt((pos[0] - origin[0])**2 + (pos[1] - origin[1])**2)


def compute_output(x, y, d):
    out = random_network(torch.tensor([x, y, d], dtype=torch.float32))
    return out.item()


def compute_output_batch(coords):
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to('cuda')

    outputs = random_network(coords_tensor)

    return outputs.detach().cpu().numpy()


def draw_single_grid(single_grid, cmap, save=False, filename="grid.png"):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    ax.matshow(single_grid, cmap=cmap)
    ax.axis("off")
    if save:
        fig.savefig(filename, dpi=25, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":

    cool_cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                  'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                  'tab20c'] + ['twilight', 'twilight_shifted', 'hsv'] + ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                                                                         'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'] + ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    width = 1080
    for _ in range(10):
        random_network = RandomNetwork().to('cuda')
        print(random_network)

        fig, ax = plt.subplots()
        width_inches = 12
        fig.set_size_inches(width_inches, width_inches)

        plt_cmap = random.choice(cool_cmaps)

        coords = [(i, j, dist_from_origin((i, j), origin=(width // 2, width // 2)))
                  for i in range(width) for j in range(width)]

        output_batch = compute_output_batch(coords)

        grid = output_batch.reshape(width, width)

        # grid = [[compute_output(i, j, dist_from_origin(
        #     (i, j), origin=(width // 2, width // 2))) for j in range(width)] for i in range(width)]

        mat = ax.matshow(grid, cmap=plt_cmap)
        ax.axis("off")
        fig.savefig(f'torch-generated/{str(uuid.uuid1())}.png',
                    dpi=width / width_inches, bbox_inches="tight")

        # plt.show()
