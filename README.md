# Image generation with CPPNs in PyTorch
## Getting started
After cloning the repo, it's suggested to setup a virtualenv and install required packages:

`pip install -r requirements.txt`

From there, you can run `torch_cppn.py`:

`python torch_cppn.py`

This should generate a random network and dispaly the generated image with matplotlib.  

## Config
The config file is declared at the top of `torch_cppn.py` and should be changed there

```
from configparser import ConfigParser

config = ConfigParser()
config_file = "config-cool-colors"
```

If you want to automatically generate and save images to the images directory, in the config change

```
[misc]

display_every_image = True
```

to
```
[misc]

display_every_image = False
``````




There are 3 default config files:

- <b>config-all</b>, which contains all of the functions and colormaps available enabled

- <b>config-basic</b>, which only uses basic activation functions and no functions for distance or the coordinates

- <b>config-cool-colors</b>, is my custom config that has a subset of the colormaps and has big_or_small_layers set to True.


### Config definitions
- <b>big_or_small_layers</b>: If set to True, layers will be EITHER big or small. This section of code shows the specifics:
```
if big_or_small_layers:
    layer_sizes = [
        random.choice(
            [random.randint(2, 25), random.randint(2, max_layer_size)]
        )
        for _ in range(num_layers)
    ]
else:
    layer_sizes = [
                random.randint(2, max_layer_size) 
                for _ in range(num_layers)
                   ]
```
