# Segmfriends
Bunch of tools and experiments for image segmentation


## Install
### Basic version
This version can run most of the clustering/segmentation algorithms included:

- `conda create -n segmFr -c abailoni -c conda-forge nifty vigra cython affogato`
- `source activate segmFr`
- `conda install GASP` (or install directly from [here](https://github.com/abailoni/GASP))
- `python setup.py install` (or `python setup.py develop` if you plan to work on the package)

### Full version
For running all deep-learning experiments, at the moment the following dependencies are required:
- `conda create -n segmFr -c abailoni -c conda-forge -c pytorch nifty vigra cython`
- `inferno` : https://github.com/elmo0082/inferno
- `firelight`: https://github.com/inferno-pytorch/firelight
- `speedrun`: https://github.com/elmo0082/speedrun 
- `neurofire`: https://github.com/abailoni/neurofire
- `ConfNets`: https://github.com/imagirom/ConfNets/tree/multi-scale-unet - (branch `multi-scale-unet`)
- `affogato`: https://github.com/constantinpape/affogato/tree/affinities-with-glia (branch `affinities-with-glia`)
- (for evaluation scores, install module `cremi`: https://github.com/constantinpape/cremi_python/tree/py3)

Coming soon: `segmfriend` conda-package
