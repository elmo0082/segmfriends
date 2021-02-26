from pathutils import get_home_dir, get_trendytukan_drive_dir, change_paths_config_file 
import sys 
sys.path.append("../ConfNets") 
import numpy as np

from speedrun import BaseExperiment, TensorboardMixin, AffinityInferenceMixin

from copy import deepcopy

import os
import torch

from segmfriends.utils.config_utils import recursive_dict_update
from segmfriends.utils.various import check_dir_and_create
from segmfriends.utils.various import writeHDF5

from neurofire.criteria.loss_wrapper import LossWrapper
from inferno.io.transform import Compose


from speedrun.py_utils import create_instance
from segmfriends.datasets.cremi import get_cremi_loader


import cremi
import vigra
import h5py
import yaml

torch.backends.cudnn.benchmark = True

class BaseCremiExperiment(BaseExperiment, AffinityInferenceMixin):
    def __init__(self, experiment_directory=None, config=None):
        super(BaseCremiExperiment, self).__init__(experiment_directory)
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        if config is not None:
            self.read_config_file(config)

        self.DEFAULT_DISPATCH = 'train'
        self.auto_setup()

        self.set_devices()

        # self.build_infer_loader()
        if "model_class" in self.get('model'):
            self.model_class = self.get('model/model_class')
        else:
            self.model_class = list(self.get('model').keys())[0]

    def set_devices(self):
        # # --------- In case of multiple GPUs: ------------
        # n_gpus = torch.cuda.device_count()
        # gpu_list = range(n_gpus)
        # self.set("gpu_list", gpu_list)
        # self.trainer.cuda(gpu_list)

        # --------- Debug on trendytukan, force to use only GPU 0: ------------
        self.set("gpu_list", [0])
        self.trainer.cuda([0])

    def build_model(self, model_config=None):
        model_config = self.get('model') if model_config is None else model_config

        assert "model_class" in model_config
        assert "model_kwargs" in model_config
        model_class = model_config["model_class"]
        model_kwargs = model_config["model_kwargs"]
        model_path = model_kwargs.pop('loadfrom', None)
        model_config = {model_class: model_kwargs}
        model = create_instance(model_config, self.MODEL_LOCATIONS)

        if model_path is not None:
            print(f"loading model from {model_path}")
            loaded_model = torch.load(model_path)["_model"]
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict)

        return model

    def inferno_build_criterion(self):
        print("Building criterion")
        loss_kwargs = self.get("trainer/criterion/kwargs", {})
        # from vaeAffs.models.losses import EncodingLoss, PatchLoss, PatchBasedLoss, StackedAffinityLoss
        loss_name = self.get("trainer/criterion/loss_name",
                             "inferno.extensions.criteria.set_similarity_measures.SorensenDiceLoss")
        loss_config = {loss_name: loss_kwargs}

        criterion = create_instance(loss_config, self.CRITERION_LOCATIONS)
        transforms = self.get("trainer/criterion/transforms")
        if transforms is not None:
            assert isinstance(transforms, list)
            transforms_instances = []
            # Build transforms:
            for transf in transforms:
                transforms_instances.append(create_instance(transf, []))
            # Wrap criterion:
            criterion = LossWrapper(criterion, transforms=Compose(*transforms_instances))

        self._trainer.build_criterion(criterion)
        self._trainer.build_validation_criterion(criterion)

    def build_train_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/train'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)

    def build_val_loader(self):
        kwargs = recursive_dict_update(self.get('loaders/val'), deepcopy(self.get('loaders/general')))
        return get_cremi_loader(kwargs)


    def build_infer_loader(self):
        kwargs = deepcopy(self.get('loaders/infer'))
        loader = get_cremi_loader(kwargs)
        return loader

    def save_infer_output(self, output):
        print("Saving....")
        if self.get("export_path") is not None:
            dir_path = os.path.join(self.get("export_path"),
                                    self.get("name_experiment", default="generic_experiment"))
        else:
            try:
                # Only works for my experiments saving on trendyTukan, otherwise it will throw an error:
                trendyTukan_path = get_trendytukan_drive_dir()
            except ValueError:
                raise ValueError("TrendyTukan drive not found. Please specify an `export_path` in the config file.")
            dir_path = os.path.join(trendyTukan_path, "projects/pixel_embeddings", self.get("name_experiment", default="generic_experiment"))
        check_dir_and_create(dir_path)
        filename = os.path.join(dir_path, "predictions_sample_{}.h5".format(self.get("loaders/infer/name")))
        print("Writing to ", self.get("inner_path_output", 'data'))
        writeHDF5(output.astype(np.float16), filename, self.get("inner_path_output", 'data'))
        print("Saved to ", filename)

        # Dump configuration to export folder:
        self.dump_configuration(os.path.join(dir_path, "prediction_config.yml"))

def connected_components_per_slice(boundary_prob, threshold=0.9):
    boundary_mask = (boundary_prob >= threshold).astype("uint8")

    boundary_mask = 1 - boundary_mask

    slices = []
    for i in range(boundary_mask.shape[0]):
        slices.append(vigra.analysis.labelMultiArray(boundary_mask[i]))
    slices = np.stack(slices)

    # make all labels distinct between slices (this is optional)
    max_label = 0
    for i in range(slices.shape[0]):
        slices[i] += max_label
        max_label = slices[i].max()
    return  slices


def cremi_score_per_slice(gt, seg, return_all_metrics=False):
    assert gt.shape == seg.shape
    gt = gt.astype(int)
    seg = seg.astype(int)
    vois_split = []
    vois_merge = []
    arands = []
    for i in range(gt.shape[0]):
        if (gt[i] != 0).sum() == 0:
            voi_split = np.nan
            voi_merge = np.nan
            arand = np.nan
        else:
            voi_split, voi_merge = cremi.voi(seg[i], gt[i])
            arand = cremi.adapted_rand(seg[i], gt[i])
        vois_split.append(voi_split)
        vois_merge.append(voi_merge)
        arands.append(arand)

    vois_split = np.array(vois_split)
    vois_merge = np.array(vois_merge)
    arands = np.array(arands)

    cremi_scores = np.sqrt(arands * (vois_split + vois_merge))

    if return_all_metrics:
        return {"voi_split": vois_split,
                "voi_merge": vois_merge,
                "adapted_rand": arands,
                "cremi_scores": cremi_scores}
    else:
        return cremi_scores

if __name__ == '__main__':
    print(sys.argv[1])

    # lots of command line argument bookkeeping
    source_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(source_path, 'configs')
    experiments_path = os.path.join(source_path, 'runs')

    # Update HCI_HOME paths:
    for i, key in enumerate(sys.argv):
        if "HCI__HOME" in sys.argv[i]:
            sys.argv[i] = sys.argv[i].replace("HCI__HOME/", get_home_dir())

    # Update RUNS paths:
    for i, key in enumerate(sys.argv):
        if "RUNS__HOME" in sys.argv[i]:
            sys.argv[i] = sys.argv[i].replace("RUNS__HOME", experiments_path)

    # sys.argv[1] = os.path.join(experiments_path, sys.argv[1])
    if '--inherit' in sys.argv:
        i = sys.argv.index('--inherit') + 1
        if sys.argv[i].endswith(('.yml', '.yaml')):
            sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]))
        else:
            sys.argv[i] = os.path.join(experiments_path, sys.argv[i])

        # update experiment name by appending _inference to experiment from which we interit
        sys.argv[1] = os.path.join(experiments_path,  "_".join([sys.argv[i], sys.argv[1], "inference"]))

        # add arguments for the model location, experiment name, export_path
        sys.argv.append("--config.model.model_kwargs.loadfrom")
        sys.argv.append(os.path.join(sys.argv[i], "best_checkpoint.pytorch"))

        sys.argv.append("--config.export_path")
        sys.argv.append(experiments_path)

        sys.argv.append("--config.name_experiment")
        sys.argv.append(sys.argv[1].split("/")[-1])

    if '--update' in sys.argv:
        i = sys.argv.index('--update') + 1
        sys.argv[i] = change_paths_config_file(os.path.join(config_path, sys.argv[i]))
    i = 0
    while True:
        if f'--update{i}' in sys.argv:
            ind = sys.argv.index(f'--update{i}') + 1
            sys.argv[ind] = change_paths_config_file(os.path.join(config_path, sys.argv[ind]))
            i += 1
        else:
            break

    # create inference experiment and obtains+saves the network output
    cls = BaseCremiExperiment()
    cls.infer()

    test_vol_config = cls._config["loaders"]["infer"]

    # run connected components
    # reload network output
    with h5py.File(os.path.join(sys.argv[1], f"predictions_sample_{test_vol_config['name']}.h5"), "r") as file:
        affs = file["data"][:]
    if affs.shape[0] == 1: # predictions for boundary prob
        boundary_probs = affs[0]
    elif affs.shape[0] == 20: # predictions for 3D offsets
        boundary_probs = (affs[1] + affs[2]) / 2
    else:  # predictions for 2D offsets
        boundary_probs = (affs[0] + affs[1]) / 2

    # compute connected components and save
    pred_seg = connected_components_per_slice(boundary_probs, cls._config["inference"]["threshold"])
    threshold_str =  "_".join(str(cls._config["inference"]["threshold"]).split("."))
    with h5py.File(os.path.join(sys.argv[1], f"predicted_segmentation_sample_{test_vol_config['name']}_{threshold_str}.h5"), "w") as file:
        file.create_dataset("pred_seg", data=pred_seg)

    # load GT
    with h5py.File(get_home_dir() + f"datasets/sample_{test_vol_config['name']}_with_boundaries.h5", "r") as file:
        gt = file["volumes"]["labels"]["neuron_ids"][:]

    # slice appropriately
    z_slice_str, _, _ = test_vol_config["volume_config"]["data_slice"].split(",")
    z_slice_1, z_slice_2 = z_slice_str.split(":")
    z_slice_1 = int(z_slice_1) if z_slice_1 != "" else None
    z_slice_2 = int(z_slice_2) if z_slice_2 != "" else None
 
    gt = gt[z_slice_1: z_slice_2]

    assert gt.shape == pred_seg.shape, f"GT and prediction should have same shape but have shapes {gt.shape} and {pred_seg.shape}"

    # compute cremi scores
    cremi_scores = cremi_score_per_slice(gt, pred_seg)

    with open(os.path.join(sys.argv[1], f"cremi_scores_{threshold_str}"), "w") as file:
        for score in cremi_scores:
            file.write(f"{score}\n")

    # omit slices without cells
    cremi_scores = cremi_scores[~np.isnan(cremi_scores)]
    print(cremi_scores)
    avg_score = np.array(cremi_scores).mean()

    print(f"CREMI score averaged across slices: {avg_score}")






