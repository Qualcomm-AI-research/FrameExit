# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Normalize, Compose, ToTensor


def __get_transforms_data_spatial(c_preprocessing):
    tfs = []
    for preprocess in vars(c_preprocessing):
        if preprocess == "resize":
            tfs += [ApplyImageTransform(Resize(c_preprocessing.resize))]

        elif preprocess == "crop_center":
            tfs += [ApplyImageTransform(CenterCrop(c_preprocessing.crop_center))]

        elif preprocess == "crop_random":
            tfs += [RandomCrop(c_preprocessing.crop_random)]

        elif preprocess == "to_tensor":
            tfs += [ApplyImageTransform(ToTensor())]

        elif preprocess == "normalize":
            tfs += [
                ApplyImageTransform(
                    Normalize(
                        c_preprocessing.normalize.mean, c_preprocessing.normalize.std
                    )
                )
            ]

        else:
            raise Exception("data pre-processing {} is unknown".format(preprocess))

    return Compose(tfs)


def __get_transforms_label(c_preprocessing):
    tfs = []
    for preprocess in vars(c_preprocessing):
        if preprocess == "video_multihot_labels":
            tfs += [VideoMultiHotLabels()]

        elif preprocess == "one_hot_encoding":
            tfs += [
                ApplyImageTransform(MultiHotEmbedding(c_preprocessing.one_hot_encoding))
            ]

        elif preprocess == "mapping":
            tfs += [ApplyImageTransform(LabelMapping(c_preprocessing.mapping))]

        else:
            raise Exception("label pre-processing {} is unknown".format(preprocess))

    return Compose(tfs)


def __get_transforms(c_data):
    tfs_spatial = __get_transforms_data_spatial(c_data.preprocessing)
    tfs_label = __get_transforms_label(c_data.preprocessing_label)

    return tfs_spatial, tfs_label


class Resize(object):
    """Resize the input PIL Image or torch.Tensor to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, size, method="bilinear"):
        assert isinstance(size, int) or len(size) == 2
        assert method in ["bilinear", "nearest"]

        self.size = size
        self.method = method

    def __call__(self, img):
        """
        Args:
            img (PIL Image or torch.Tensor): Image to be scaled.

        Returns:
            PIL Image or torch.Tensor: Rescaled image.
        """
        assert isinstance(img, (torch.Tensor, Image.Image))

        if isinstance(img, torch.Tensor):
            return self.__tensor_call__(img, self.method)
        else:
            return TF.resize(
                img,
                self.size,
                Image.BILINEAR if self.method == "bilinear" else Image.NEAREST,
            )

    def __tensor_call__(self, tensor, method="bilinear"):
        if isinstance(self.size, int):
            c, h, w = tensor.size()
            assert c == 3
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return tensor
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return F.interpolate(
                    tensor.unsqueeze(0), (oh, ow), mode=method, align_corners=False
                )[0]
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return F.interpolate(
                    tensor.unsqueeze(0), (oh, ow), mode=method, align_corners=False
                )[0]
        else:
            oh, ow = self.size
            return F.interpolate(
                tensor.unsqueeze(0), (oh, ow), mode=method, align_corners=False
            )[0]

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, method={1})".format(
            self.size, self.method
        )


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        assert isinstance(size, int) or len(size) == 2
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image or torch.Tensor): Image to be cropped.

        Returns:
            img (PIL Image or torch.Tensor): Cropped image.
        """

        assert isinstance(img, (torch.Tensor, Image.Image))

        if isinstance(img, torch.Tensor):
            return self.__tenosr_call__(img)
        else:
            return TF.center_crop(img, self.size)

    def __tenosr_call__(self, tensor):
        if isinstance(self.size, int):
            output_size = (self.size, self.size)
        else:
            output_size = tuple(self.size)

        c, h, w = tensor.size()
        assert c == 3

        th, tw = output_size
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))

        return tensor[:, i : i + th, j : j + tw]

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class ApplyImageTransform(object):
    """
    Apply an image transform to all video frames
    """

    def __init__(self, transform):
        """
        :param transform: a transform object from "torchvision.transforms.transforms.*" such as Resize or GrayScale

        """
        self.transform = transform

    def __call__(self, frames):
        """
        Parameters
        ----------
        frames: list of frames in PIL.Image or numpy.ndarray format

        Returns
        -------
        List of transformed frames

        """
        transformed = [self.transform(frame) for frame in frames]
        return transformed

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.transform)


class RandomCrop(object):
    """
    Extract random crop at the same location over frames
    """

    def __init__(self, size):
        """

        Parameters
        ----------
        size: crop size in format (h, w)
        """

        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, frames):
        """

        Parameters
        ----------
        frames list of frames in PIL.Image or numpy.ndarray format (HxWxC)

        Returns
        -------
        List of cropped frames in the same format as input (PIL.Image or numpy.ndarray)

        """

        assert isinstance(frames[0], (torch.Tensor, Image.Image, np.ndarray))

        if isinstance(frames[0], torch.Tensor):
            return self.__tensor_call__(frames)
        else:
            crop_h, crop_w = self.size
            if isinstance(frames[0], np.ndarray):
                frames = [Image.fromarray(frame) for frame in frames]

            frame_w, frame_h = frames[0].size
            crop_left = random.randint(0, frame_w - crop_w)
            crop_top = random.randint(0, frame_h - crop_h)
            transformed = [
                frame.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))
                for frame in frames
            ]

            if isinstance(frames[0], np.ndarray):
                transformed = [np.asarray(x) for x in transformed]

            return transformed

    def __tensor_call__(self, frames):
        c, h, w = frames[0].size()
        assert c == 3

        crop_h, crop_w = self.size
        if crop_w > w:
            frames = ApplyImageTransform(Resize(size=crop_w))(frames)
            c, h, w = frames[0].size()

        if crop_h > h:
            frames = ApplyImageTransform(Resize(size=crop_h))(frames)
            c, h, w = frames[0].size()

        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)

        transformed = [
            frame[:, top : top + crop_h, left : left + crop_w] for frame in frames
        ]
        return transformed

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.size)


class MultiHotEmbedding(object):
    """
    Create a Multi-hot label vector of length num_classes for multi label classification
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, label: list):
        """
        Parameters
        ----------
        label: list
            list of class indices to be embedded

        Returns
        ----------
         torch.tensor: label tensor of length num_classes, where each element from label list is one-hot embedded
        ----------
        """

        return torch.zeros(self.num_classes).scatter(
            0, torch.tensor(label, dtype=torch.long), torch.tensor(1)
        )

    def __repr__(self):
        return self.__class__.__name__ + f"for {self.num_classes} classes"


class VideoMultiHotLabels(object):
    """
    Extracts video level labels. Assumes that labels of the first frame in the video is the same as the labels of all
    other frames and returns that list of labels.
    """

    def __init__(self):
        pass

    def __call__(self, context: dict):
        labels = context[0]["label"]
        return [labels]


class LabelMapping(object):
    """
    map labels using a given mapping file each line two space-separated values e.g.  12 0
    """

    def __init__(self, mapping_path):
        self.mapper = {
            int(line.strip().split(" ")[0]): int(line.strip().split(" ")[1])
            for line in open(mapping_path)
        }

    def __call__(self, label):
        label = [self.mapper[line] for line in label]
        return label

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
