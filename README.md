# Image generation with CPPNs in PyTorch
## Getting started
After cloning the repo, it's suggested to setup a virtualenv and install required packages:

`pip install -r requirements.txt`

From there, you can run `torch_cppn.py`:

`python torch_cppn.py`

This should generate a random network and display the generated image with matplotlib

<div style="display: flex; justify-content: space-between;">
    <img src="cool_images\02c9e679-bef9-11ee-b087-10a56209f40c.png" alt="Image 1" width="20%">
    <img src="cool_images\ab535087-bf8d-11ee-8cc0-10a56209f40c.png" alt="Image 2" width="20%">
    <img src="cool_images\e78b0119-bf97-11ee-a1bd-10a56209f40c.png" alt="Image 3" width="20%">
    <img src="cool_images\8133bed0-bf25-11ee-b45e-10a56209f40c - Copy.png" alt="Image 4" width="20%">
    <img src="cool_images\193aaea8-befb-11ee-94a7-10a56209f40c - Copy.png" alt="Image 4" width="20%">

    
</div>

## Config
The config file is declared at the top of `torch_cppn.py` and can be changed there 

```
from configparser import ConfigParser

config = ConfigParser()
config_file = "config-cool-colors"
```
### Editing config and viewing images
If you want to automatically generate and save images to the images directory without displaying them, in the config change

```
[misc]

display_every_image = True
```
to
```
[misc]

display_every_image = False
``````
If set to True, the next image won't be generated until you close the display, so you can edit the config file while the display is open and the changes will be reflected in the next image generated. This hopefully can make customizing images quicker and easier

### Config files
There are 4 default config files:

- <b>config-all</b>, which contains all of the functions and colormaps available enabled

- <b>config-all-no-cuda</b>, which contains all of the functions and colormaps available enabled, but will run only on the CPU (much slower, so everything is smaller)

- <b>config-basic</b>, which only uses basic activation functions and no functions for distance or the coordinates

- <b>config-cool-colors</b>, is my custom config that has a subset of the colormaps and has big_or_small_layers set to True.



### Config definitions
- <b>big_or_small_layers</b>: If set to True, layers will be EITHER big or small. This section of code shows the specifics:
```
if big_or_small_layers:
    layer_sizes = [
        random.choice(
            [random.randint(2, 25), random.randint(min(max_layer_size, 65), max_layer_size)]
        )
        for _ in range(num_layers)
    ]
else:
    layer_sizes = [
                random.randint(2, max_layer_size) 
                for _ in range(num_layers)
                   ]
```
## Images

### Colormaps
The colormaps used are from matplotlib and can be found here:
https://matplotlib.org/stable/gallery/color/colormap_reference.html


Examples of each of the colormaps used can be found in the colormap_examples directory in the folder with the respective colormap's name

Some of my personal favorite images are in the cool_images directory

