# Segmfriends
Bunch of tools and experiments for image segmentation

### Full version
For running all deep-learning experiments, at the moment the following dependencies are required:
- `conda create -n segmFr -c abailoni -c conda-forge -c pytorch nifty vigra cython yaml opencv`
- `inferno` : https://github.com/elmo0082/inferno/tree/matplotlib-plotting - (branch `matplotlib-plotting`)
- `firelight`: https://github.com/inferno-pytorch/firelight
- `speedrun`: https://github.com/elmo0082/speedrun/tree/matplotlib-plotting - (branch `matplotlib-plotting`)
- `neurofire`: https://github.com/abailoni/neurofire
- `ConfNets`: https://github.com/imagirom/ConfNets/tree/multi-scale-unet - (branch `multi-scale-unet`)
- `affogato`: https://github.com/constantinpape/affogato/tree/affinities-with-glia (branch `affinities-with-glia`)
- (for evaluation scores, install module `cremi`: https://github.com/constantinpape/cremi_python/tree/py3)


### Building affogato

The package ```affogato``` has to be built using ```conda build .``` in the root directory.
You can find the compiled package in your environment directory under ```<your conda directory>/conda-bld/linux-64```.
The conda package can be installed with ```conda install <affogato package name>``` .
