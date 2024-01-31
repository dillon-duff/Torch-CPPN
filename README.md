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

If you want to automatically generate and save images to the images directory, change

```
[misc]

display_every_image = True
```

to
```
[misc]

display_every_image = False
``````




There are 3 default config files, config-all which contains all of the functions and colormaps available, config-basic which only uses basic activation functions and no functions for distance or the coordinates, and config-cool-colors is my custom config that has a subset of the colormaps.

