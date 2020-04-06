import numpy as np
from scipy.ndimage import zoom

try:
    from inferno.io.transform import Compose, Transform
except ImportError:
    raise ImportError("Volume transforms requires inferno")

from ..utils.various import parse_data_slice


class DuplicateGtDefectedSlices(Transform):
    def __init__(self, defects_label=2, ignore_label=0, **super_kwargs):
        self.defects_label = defects_label
        self.ignore_label = ignore_label
        super(DuplicateGtDefectedSlices, self).__init__(**super_kwargs)

    def batch_function(self, batch):
        assert len(batch) == 2, "This transform expects two tensors: raw, gt_segm"
        targets = batch[1]

        if targets.ndim == 4 and targets.shape[0] != 1:
            assert targets.shape[0] == 2, "Targets can have max two channels: gt labels and extra masks"

            defect_mask = targets[1] == self.defects_label

            # On the first slice we should never have defects:
            if defect_mask[0].max():
                print("WARNING: defects on first slice!!!")
                # In this special case, we set GT of the first slice to ignore label:
                targets[:, 0] = self.ignore_label

            # For each defect, get GT from the previous slice
            for z_indx in range(1, defect_mask.shape[0]):
                # Copy GT:
                targets[0, z_indx][defect_mask[z_indx]] = targets[0, z_indx-1][defect_mask[z_indx]]
                # Copy masks:
                targets[1, z_indx][defect_mask[z_indx]] = targets[1, z_indx-1][defect_mask[z_indx]]
        else:
            assert targets.ndim == 3, "Targets should be either 3 or 4 dimensional tensor"

        return (batch[0], targets)


class MergeExtraMasks(Transform):
    """
    Merge extra_masks loaded from file (with masks of boundary pixels, glia segments and defected slices in the dataset)
    with the mask of slices modified during augmentation (which now contain artifacts, are black out, etc...)
    """
    def __init__(self, defects_label=2, **super_kwargs):
        self.defects_label = defects_label
        super(MergeExtraMasks, self).__init__(**super_kwargs)

    def batch_function(self, batch):
        extra_masks = None
        if len(batch) == 3:
            raw_data, gt, extra_masks = batch
        else:
            assert len(batch) == 2, "This transform requires either two or three tensors: raw, GT_segm, (extra_masks)"
            raw_data, gt = batch
        if isinstance(raw_data, (tuple, list)):
            assert len(raw_data) == 2, "Raw data should be either one tensor or a list of two tensors"
            # Create an empty tensor, if None:
            extra_masks = np.zeros_like(gt) if extra_masks is None else extra_masks
            # Combine defects (in the original data and from augmentation):
            raw_data, augmented_defected_mask = raw_data
            extra_masks[augmented_defected_mask.astype('bool')] = self.defects_label

        if extra_masks is not None:
            # Concatenate labels in one tensor:
            gt = np.stack([gt,extra_masks])

        return (raw_data, gt)


class ReplicateTensorsInBatch(Transform):
    """
    Creates replicates of the tensors in the batch (used to create multi-scale inputs).
    """
    def __init__(self,
                 indices_to_replicate,
                 **super_kwargs):
        super(ReplicateTensorsInBatch, self).__init__(**super_kwargs)
        self.indices_to_replicate = indices_to_replicate

    def batch_function(self, batch):
        new_batch = []
        for indx in self.indices_to_replicate:
            assert indx < len(batch)
            new_batch.append(np.copy(batch[indx]))
        return new_batch


class DownSampleAndCropTensorsInBatch(Transform):
    def __init__(self,
                 ds_factor=(1, 2, 2),
                 crop_factor=None,
                 crop_slice=None,
                 order=None,
                 **super_kwargs):
        """
        :param ds_factor: If factor is 2, then downscaled to half-resolution
        :param crop_factor: If factor is 2, the central crop of half-size is taken
        :param crop_slice: Alternatively, a string can be passed representing the desired crop
        :param order: downscaling order. By default, it is set to 3 for 'float32'/'float64' and to 0 for
                    integer/boolean data types.
        """
        super(DownSampleAndCropTensorsInBatch, self).__init__(**super_kwargs)
        self.order = order
        self.ds_factor = ds_factor
        if crop_factor is not None:
            assert isinstance(crop_factor, (tuple, list))
            # assert crop_slice is None
        if crop_slice is not None:
            # assert crop_slice is not None
            assert isinstance(crop_slice, str)
            crop_slice = parse_data_slice(crop_slice)
        self.crop_factor = crop_factor
        self.crop_slice = crop_slice

    def volume_function(self, volume):
        # Downscale the volume:
        downscaled = volume
        if (np.array(self.ds_factor) != 1).any():
            order = self.order
            if self.order is None:
                if volume.dtype in [np.dtype('float32'), np.dtype('float64')]:
                    order = 3
                elif volume.dtype in [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'), np.dtype('int64')]:
                    order = 0
                else:
                    raise ValueError
            downscaled = zoom(volume, tuple(1./fct for fct in self.ds_factor), order=order)

        # Crop:
        cropped = downscaled
        if self.crop_factor is not None:
            shape = downscaled.shape
            cropped_shape = [int(shp/crp_fct) for shp, crp_fct in zip(shape, self.crop_factor)]
            offsets = [int((shp-crp_shp)/2) for shp, crp_shp in zip(shape, cropped_shape)]
            crop_slc = tuple(slice(off, off+crp_shp) for off, crp_shp in zip(offsets, cropped_shape))
            cropped = downscaled[crop_slc]

        if self.crop_slice is not None:
            # Crop using the passed crop_slice:
            cropped = cropped[self.crop_slice]
        return cropped

    def apply_to_torch_tensor(self, tensor):
        assert tensor.ndimension() == 5
        assert (np.array(self.ds_factor) == 1).all(), "Atm not implemented (Zoom not applicable to tensors)"

        # Crop:
        cropped = tensor
        if self.crop_factor is not None:
            shape = tensor.shape[-3:]
            cropped_shape = [int(shp/crp_fct) for shp, crp_fct in zip(shape, self.crop_factor)]
            offsets = [int((shp-crp_shp)/2) for shp, crp_shp in zip(shape, cropped_shape)]
            crop_slc = (slice(None), slice(None)) + tuple(slice(off, off+crp_shp) for off, crp_shp in zip(offsets, cropped_shape))
            cropped = tensor[crop_slc]

        if self.crop_slice is not None:
            # Crop using the passed crop_slice:
            cropped = cropped[(slice(None), slice(None)) + self.crop_slice]

        return cropped


class CheckBatchAndChannelDim(Transform):
    def __init__(self, dimensionality, *super_args, **super_kwargs):
        super(CheckBatchAndChannelDim, self).__init__(*super_args, **super_kwargs)
        self.dimensionality = dimensionality

    def batch_function(self, batch):
        output_batch = []
        for tensor in batch:
            if tensor.ndimension() == self.dimensionality:
                output_batch.append(tensor.unsqueeze(0).unsqueeze(0))
            elif tensor.ndimension() == self.dimensionality + 1:
                output_batch.append(tensor.unsqueeze(0))
            elif tensor.ndimension() == self.dimensionality + 2:
                output_batch.append(tensor)
            else:
                raise ValueError
        return tuple(output_batch)