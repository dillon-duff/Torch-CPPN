import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import uuid

import torch
import torch.nn as nn
import random

from configparser import ConfigParser

config = ConfigParser()
config_file = "config-cool-colors"


def read_config():
    global min_layers, max_layers, max_layer_size, act_funcs, color_maps, image_width, dist_funcs, max_vertical_subdivisions_distance, max_horizontal_subdivisions_distance, max_vertical_subdivisions_coords, max_horizontal_subdivisions_coords, coord_funcs, display_every_image, big_or_small_layers
    config.read("config-basic")
    min_layers = int(config["network"]["min_layers"])
    max_layers = int(config["network"]["max_layers"])
    max_layer_size = int(config["network"]["max_layer_size"])
    big_or_small_layers = config["network"]["big_or_small_layers"].lower() == "true"
    act_funcs = config["network"]["act_funcs"].split()
    

    color_maps = config["drawing"]["color_maps"].split()
    image_width = int(config["drawing"]["image_width"])

    dist_funcs = config["distance_functions"]["dist_funcs"].split()
    max_vertical_subdivisions_distance = int(
        config["distance_functions"]["max_vertical_subdivisions_distance"]
    )
    max_horizontal_subdivisions_distance = int(
        config["distance_functions"]["max_horizontal_subdivisions_distance"]
    )
    max_vertical_subdivisions_coords = int(
        config["distance_functions"]["max_vertical_subdivisions_coords"]
    )
    max_horizontal_subdivisions_coords = int(
        config["distance_functions"]["max_horizontal_subdivisions_coords"]
    )

    coord_funcs = config["coordinate_funcs"]["coord_funcs"].split()

    display_every_image = config["misc"]["display_every_image"].lower() == "true"


class RandomNetwork(nn.Module):
    def __init__(self):
        super(RandomNetwork, self).__init__()

        num_layers = random.randint(min_layers, max_layers)

        if big_or_small_layers:
            layer_sizes = [random.choice([random.randint(2, 25), random.randint(2, max_layer_size)]) for _ in range(num_layers)]
        else:
            layer_sizes = [random.randint(2, max_layer_size) for _ in range(num_layers)]

        input_size = 3

        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(random.choice(active_act_funcs))
            input_size = size

        layers.append(nn.Linear(layer_sizes[-1], 1))
        layers.append(random.choice(active_act_funcs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def distance_from_center(x, y, width, height, **kwargs):
    return np.sqrt((x - (width // 2)) ** 2 + (y - (height // 2)) ** 2)


def distance_from_dynamic_spiral_center(
    x, y, width, height, spiral_density=0.1, **kwargs
):
    center_x, center_y = width / 2, height / 2

    rel_x, rel_y = x - center_x, y - center_y

    angle = np.arctan2(rel_y, rel_x)
    radius = np.hypot(rel_x, rel_y)

    spiral_x = center_x + spiral_density * angle * np.cos(angle)
    spiral_y = center_y + spiral_density * angle * np.sin(angle)

    distance = np.hypot(spiral_x - x, spiral_y - y)

    return distance


def distance_from_nearest_subdivided_quadrant_center(
    x, y, width, height, subdiv_x=1, subdiv_y=1, **kwargs
):
    square_width = width / subdiv_x
    square_height = height / subdiv_y

    nearest_center_x = (int(x / square_width) + 0.5) * square_width
    nearest_center_y = (int(y / square_height) + 0.5) * square_height

    distance = np.hypot(nearest_center_x - x, nearest_center_y - y)

    return distance


def distance_from_nearest_center(x, y, *args, **kwargs):
    centers = kwargs.get("centers", [(0, 0)])
    centers_array = np.array(centers)

    dx = centers_array[:, 0] - x
    dy = centers_array[:, 1] - y

    distances = np.hypot(dx, dy)

    return np.min(distances)


def distance_from_concentric_circles(x, y, width, height, circle_spacing=10, **kwargs):
    center_x, center_y = width / 2, height / 2
    distance_to_center = np.hypot(x - center_x, y - center_y)

    nearest_circle_radius = round(distance_to_center / circle_spacing) * circle_spacing
    distance = abs(distance_to_center - nearest_circle_radius)

    return distance


def distance_from_nearest_edge(x, y, width, height, **kwargs):
    distance_to_left_edge = x
    distance_to_right_edge = width - x
    distance_to_top_edge = y
    distance_to_bottom_edge = height - y

    return min(
        distance_to_left_edge,
        distance_to_right_edge,
        distance_to_top_edge,
        distance_to_bottom_edge,
    )


def distance_from_diagonal(x, y, width, height, **kwargs):
    # Equation of the diagonal y = x * (height / width)
    diagonal_slope = height / width
    distance = abs((diagonal_slope * x - y) / np.sqrt(diagonal_slope**2 + 1))

    return distance


def compute_output_batch(coords, batch_size=40000):
    outputs = []
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        coords_tensor = torch.tensor(batch_coords, dtype=torch.float32).to("cuda")
        batch_output = random_network(coords_tensor)
        outputs.append(batch_output.detach().cpu().numpy())
        del coords_tensor
        torch.cuda.empty_cache()

    return np.concatenate(outputs, axis=0)


def coords_to_sub_div_coords(x, y, **kwargs):
    width = height = kwargs.get("width", 1)
    subdiv_x, subdiv_y = kwargs["subdiv_x"], kwargs["subdiv_y"]
    sub_width = width / subdiv_x
    sub_height = height / subdiv_y

    mod_x = x % sub_width
    mod_y = y % sub_height

    return mod_x, mod_y


def coords_identity(x, y, **kwargs):
    return x, y


def coords_to_polar(x, y, **kwargs):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta


def coords_wave_distortion(x, y, **kwargs):
    frequency = kwargs.get("frequency", 5)
    amplitude = kwargs.get("amplitude", 0.5)

    x_new = x + amplitude * np.sin(frequency * y)
    y_new = y + amplitude * np.sin(frequency * x)

    return x_new, y_new


def coords_spiral_twist(x, y, **kwargs):
    twist_rate = kwargs.get("twist_rate", 1)
    distance = np.hypot(x, y)
    angle = np.arctan2(y, x) + twist_rate * distance

    x_new = distance * np.cos(angle)
    y_new = distance * np.sin(angle)

    return x_new, y_new


def coords_scale_translate(x, y, **kwargs):
    scale_x = kwargs.get("scale_x", 1)
    scale_y = kwargs.get("scale_y", 1)
    translate_x = kwargs.get("translate_x", 0)
    translate_y = kwargs.get("translate_y", 0)

    x_new = x * scale_x + translate_x
    y_new = y * scale_y + translate_y

    return x_new, y_new


coord_func_dict = {
    "wave_distortion": coords_wave_distortion,
    "to_polar": coords_to_polar,
    "scale_translate": coords_scale_translate,
    "identity": coords_identity,
    "to_sub_div_coords": coords_to_sub_div_coords,
}

dist_func_dict = {
    "from_center": distance_from_center,
    "from_nearest_center": distance_from_nearest_center,
    "from_dynamic_spiral_center": distance_from_dynamic_spiral_center,
    "from_nearest_subdivided_quadrant_center": distance_from_nearest_subdivided_quadrant_center,
    "from_concentric_circles": distance_from_concentric_circles,
    "from_diagonal": distance_from_diagonal,
    "from_nearest_edge": distance_from_nearest_edge,
}


def compute_output(x, y, d):
    out = random_network(torch.tensor([x, y, d], dtype=torch.float32))
    return out.item()


def compute_output_batch(coords):
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to("cuda")

    outputs = random_network(coords_tensor)

    return outputs.detach().cpu().numpy()


if __name__ == "__main__":
    for _ in range(5):

        read_config()
        active_dist_funcs = [dist_func_dict[func] for func in dist_funcs]
        active_coord_funcs = [coord_func_dict[func] for func in coord_funcs]

        all_act_funcs = [
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

        act_func_dict = {str(act).split("(")[0].lower(): act for act in all_act_funcs}

        active_act_funcs = [act_func_dict[func] for func in act_funcs]

        best_cmaps = [
            "Paired",
            "Pastel1",
            "Pastel2",
            "Set1",
            "Set2",
            "Set3",
            "tab10",
            "tab10",
            "tab20",
            "tab20b",
            "tab20c",
            "twilight",
        ]

        width = image_width
        params = {
            "cmap": random.choice(color_maps),
            "x_subsections": random.randint(1, max_horizontal_subdivisions_distance),
            "y_subsections": random.randint(1, 5),
            "dist_func": random.choice(active_dist_funcs),
            "coords_func": random.choice(active_coord_funcs),
        }

        num_random_centers = params["x_subsections"] * params["y_subsections"]
        random_centers = [
            (random.randrange(width), random.randrange(width))
            for _ in range(num_random_centers)
        ]

        dist_kwargs = {
            "subdiv_x": random.randint(1, max_vertical_subdivisions_distance),
            "subdiv_y": random.randint(1, max_horizontal_subdivisions_distance),
            "spiral_density": random.random(),
            "centers": random_centers,
        }

        coords_kwargs = {
            "subdiv_x": random.randint(1, max_vertical_subdivisions_coords),
            "subdiv_y": random.randint(1, max_horizontal_subdivisions_coords),
            "width": width,
        }

        random_network = RandomNetwork().to("cuda")

        fig, ax = plt.subplots()
        width_inches = 12
        fig.set_size_inches(width_inches, width_inches)

        x_subsections = params["x_subsections"]
        y_subsections = params["y_subsections"]

        coords = [
            (
                *params["coords_func"](i, j, **coords_kwargs),
                params["dist_func"](i, j, width, width, **dist_kwargs),
            )
            for i in range(width)
            for j in range(width)
        ]

        output_batch = compute_output_batch(coords)

        grid = output_batch.reshape(width, width)

        mat = ax.matshow(grid, cmap=params["cmap"])
        ax.axis("off")
        uid = str(uuid.uuid1())
        fname = f"images/{uid}"

        fig.savefig(f"{fname}.png", dpi=width / width_inches, bbox_inches="tight")
        if display_every_image:
            plt.show()
        plt.close()
        print(random_network)
        print(params)
