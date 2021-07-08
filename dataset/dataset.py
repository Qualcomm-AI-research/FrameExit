# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import glob
import json
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from dataset.transform import __get_transforms


def get_dataloader(c_data):
    tfs_spatial, tfs_label = __get_transforms(c_data)

    read_label_func = None
    if c_data.name == "activitynet1.3":
        read_label_func = read_label_activitynet
    elif c_data.name == "minikinetics":
        read_label_func = read_label_minikinetics

    dataset = VideoLoader(
        c_data.path_split,
        c_data.path_frame,
        c_data.path_label,
        c_data.path_classid,
        read_label_func=read_label_func,
        transform_spatial=tfs_spatial,
        transform_label=tfs_label,
        clip_length=c_data.num_frames,
    )

    dataloader = DataLoader(
        dataset,
        pin_memory=True,
        batch_size=1,
        shuffle=c_data.shuffle,
        num_workers=c_data.num_workers,
    )

    return dataloader


class VideoLoader(data.Dataset):
    def __init__(
        self,
        path_split,
        path_frames,
        path_label,
        path_classid,
        read_label_func=None,
        clip_length=10,
        transform_spatial=None,
        transform_label=None,
    ):
        super().__init__()
        self.path_split = path_split
        self.path_frames = path_frames
        self.path_label = path_label
        self.path_classid = path_classid

        self.transform_spatial = transform_spatial
        self.transform_label = transform_label
        self.clip_length = clip_length

        video_keys = [line.strip() for line in open(path_split, "r")]
        self.number_of_videos = len(video_keys)

        # label
        self.video_context = read_label_func(self.path_label)
        class_id = {line.strip(): i for i, line in enumerate(open(path_classid, "r"))}
        for vname, context in self.video_context.items():
            if vname in video_keys:
                self.video_context[vname]["label"] = [
                    class_id[line] for line in context["label"]
                ]

        # frame
        video_info = {}
        for vname in video_keys:
            if os.path.isdir(os.path.join(path_frames, vname)):
                frame_names = glob.glob(os.path.join(path_frames, vname) + "/*.jpeg")
                if len(frame_names) > 0:
                    video_info[vname] = frame_names

        self.clip_indices, self.clip_names = sample_frames_uniform(
            video_info, clip_length=self.clip_length
        )

    def __getitem_label__(self, frame_list, vname):
        metadata = {"frame_ids": [], "labels": []}
        for frame_name in frame_list:
            metadata["labels"].append(self.video_context[vname])
            metadata["frame_ids"].append(frame_name)

        labels = metadata["labels"]
        if self.transform_label is not None:
            labels = self.transform_label(labels)

        labels = torch.stack(labels)

        return labels, metadata

    def __getitem__(self, index):
        frame_list = self.clip_indices[index]
        video_name = self.clip_names[index]

        clip = []
        for frame_name in frame_list:
            frame = Image.open(frame_name)
            clip.append(frame)

        if self.transform_spatial is not None:
            clip = self.transform_spatial(clip)

        clip = torch.stack(clip)
        label, metadata = self.__getitem_label__(frame_list, video_name)

        return clip, (label, metadata)

    def __len__(self):
        return len(self.clip_indices)


def sample_frames_uniform(video_indices, clip_length):
    """
    selects one clip of length clip_length uniformly from a video
    """
    splits, split_names = [], []
    for video_name, frame_list in sorted(video_indices.items()):
        video_frames = np.array(sorted(frame_list))
        video_length = len(video_frames)

        indices = np.clip(
            np.linspace(0, video_length, clip_length), 0, video_length - 1
        ).astype("int")

        split = video_frames[indices].tolist()

        splits.append(split)
        split_names.append(video_name)

    return splits, split_names


def read_label_activitynet(path_labelfile):
    # read label file
    with open(path_labelfile) as json_file:
        dataset = json.load(json_file)
    dataset = dataset["database"]

    metadata = {}
    for key, val in dataset.items():
        vkey = "v_%s.mp4" % key
        labels = [ann["label"] for ann in dataset[key]["annotations"]]
        segments = [ann["segment"] for ann in dataset[key]["annotations"]]
        cur_cntx = {"label": labels, "segment": segments, "set": dataset[key]["subset"]}
        metadata[vkey] = cur_cntx

    return metadata


def read_label_minikinetics(path_labelfile):
    # [1:] to skip csv header
    data = [x.strip() for x in open(path_labelfile, "r")][1:]

    metadata = {}
    for i, line in enumerate(data):
        label, ytname, b, e, split, cc = line.strip().split(",")
        if label[0] == '"':
            label = label[1:-1]
        vkey = "%s_%s_%s.mp4" % (
            ytname,
            b.split(".")[0].zfill(6),
            e.split(".")[0].zfill(6),
        )
        cur_cntx = {"label": [label], "cc": cc}
        metadata[vkey] = cur_cntx

    return metadata
