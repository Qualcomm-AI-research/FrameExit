# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MaxPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1)), dim=1)

        return x.max(dim=1)[0]


def execute_exiting(func, out_ave, pre_exit_features=None):
    if pre_exit_features is not None:
        exit_door_val = func(out_ave, prev_features=pre_exit_features[-1])
    else:
        exit_door_val = func(out_ave)

    return exit_door_val


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_neurons=4096):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = [num_neurons]

        layers = []
        dim_input = input_dim
        for dim_output in self.num_neurons:
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(nn.BatchNorm1d(dim_output))
            layers.append(nn.ReLU())
            dim_input = dim_output

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x


def get_torchvision_model(
    name, pretrained=True, requires_grad=False, truncate_modules=None
):
    torchvision_models = models
    if "." in name:
        prefix, name = name.split(".")[0], name.split(".")[1]
        assert prefix in vars(torchvision_models).keys()
        torchvision_models = vars(torchvision_models)[prefix]
    assert name in vars(torchvision_models).keys()

    if name == "inception_v3":
        model = vars(torchvision_models)[name](pretrained=pretrained, aux_logits=False)
    else:
        model = vars(torchvision_models)[name](pretrained=pretrained)
    if truncate_modules is not None:
        model = torch.nn.Sequential(*list(model.children())[:truncate_modules])
    for param in model.parameters():
        param.requires_grad = requires_grad

    if not requires_grad:
        model.eval()

    return model


def get_base_model(name, config):
    truncate_modules = (
        config.model.backbone.truncate_modules
        if config.model.backbone.get("truncate_modules")
        else None
    )
    if name is None:
        return None
    if "torchvision" in name.lower():
        model_name = name.split(".", 1)[-1]
        model = get_torchvision_model(
            name=model_name,
            pretrained=config.model.backbone.pretrained,
            requires_grad=config.model.backbone.requires_grad,
            truncate_modules=truncate_modules,
        )
    else:
        raise Exception("couldn't find %s as a model name" % name)

    return model


class ExitingGate(nn.Module):
    def __init__(self, in_planes):
        super(ExitingGate, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 128, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(128, 1, bias=True)

    def forward(self, x, force_hard=True, prev_features=None):
        x0, x1 = x[0], x[1]
        x0 = F.relu(self.bn1(self.conv1(x0)))
        x0 = F.relu(self.bn2(self.conv2(x0)))
        x0 = torch.flatten(x0, 1)
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = torch.flatten(x1, 1)
        x = torch.cat([x0, x1], dim=1)
        out = self.linear(x)
        out = self.sigmoid(out)
        out[out >= 0.5] = 1
        out[out < 0.5] = 0

        return out


class AdaptiveBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone_type = config.model.backbone.type
        self.backbone_name = config.model.backbone.name

        # base model
        self.backbone = get_base_model(config.model.backbone.name, config)
        model_output_dim = config.model.backbone.output_dim

        # fully connected layer
        num_neurons = 4096
        self.mlp = MultiLayerPerceptron(
            input_dim=model_output_dim, num_neurons=num_neurons
        )
        self.model_output_dim = num_neurons

        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))  # in case of frame as input
        self.num_frames = config.data.num_frames

    def forward(self, x):
        b = x.size(0)
        seq_len = x.size(1)

        # Mix batch and T
        x = x.view((b * seq_len,) + x.size()[2:])

        x = self.backbone(x)

        # separate batch and T
        if len(x.shape) > 2:
            x = self.avg_pool_2d(x)  # remove spatial dim
        x = x.view(
            (
                b,
                seq_len,
            )
            + x.size()[1:]
        )
        x = x.flatten(start_dim=2)

        # fc layer
        x = x.view(b * seq_len, np.prod(x.size()[2:]))
        x = self.mlp(x)
        x = x.view(b, seq_len, -1)

        return x


def threshold_selection(y_t, threshold=0.99):
    y_t_probs = torch.nn.Softmax(dim=1)(y_t)
    exit_door = torch.max(y_t_probs, dim=1)[0] > threshold

    return False if exit_door[0] == 0 else True


class ConditionalFrameExitInferenceModel(AdaptiveBase):
    def __init__(self, config):
        super().__init__(config)
        self.num_frames = config.data.num_frames
        self.num_class = config.model.num_class
        self.first_threshold = config.model.first_threshold
        self.max_pooling = MaxPooling()
        self.exit_selector = nn.ModuleList()
        self.exit_door = None
        self.exited_classifiers = None
        self.classifiers = nn.ModuleList()
        for m in range(self.num_frames):
            self.classifiers.append(nn.Linear(self.model_output_dim, self.num_class))
            if m > 0:
                self.exit_selector.append(ExitingGate(4096))

    def gate_selection(self, idx, y_t):
        exit_door = execute_exiting(self.exit_selector[idx], y_t)

        return True if exit_door[0] == 0 else False

    def forward(self, x, z_previous=None, t=torch.tensor(0)):
        y_t = None
        z_t = super().forward(x)
        z_t = z_t.squeeze(dim=1)
        b = z_t.shape[0]
        self.exit_door = torch.zeros(
            [x.shape[0], len(self.classifiers)], device=x.device
        )
        if t > 0:
            z_t = self.max_pooling.forward(z_t, z_previous)

        # for the first frame, we use a simple confidence threshold to exit
        if t == 0:
            z_t = torch.flatten(z_t, start_dim=1)
            y_t = self.classifiers[t](z_t)
            exited = threshold_selection(y_t, threshold=self.first_threshold)
        elif t < self.num_frames - 1:
            exited = self.gate_selection(
                t - 1, [z_t.view(b, -1, 1, 1), z_previous.view(b, -1, 1, 1)]
            )
        else:
            exited = True

        if exited:
            if t > 0:
                z_t = torch.flatten(z_t, start_dim=1)
                y_t = self.classifiers[t](z_t)
            self.exited_classifiers = t + 1
            self.exit_door[0][t] = 1
            return y_t, None
        else:
            return None, z_t
