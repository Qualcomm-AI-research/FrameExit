# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
from ignite.metrics import Metric


def softmax(logit):
    e_x = np.exp(logit - np.max(logit))
    return e_x / e_x.sum()


def sigmoid(logit):
    return 1 / (1 + np.exp(-logit))


class Hitat1(Metric):
    """
    Performs a local (numpy) calculation of the hit at one.
    """

    def __init__(
        self, aggregation="max", output_transform=lambda x: x, activation_fn=None
    ):
        self.predictions_dict = {}
        self.actuals_dict = {}
        self.aggregation = np.max if aggregation == "max" else np.mean
        self.apply_activation = (
            softmax
            if activation_fn == "softmax"
            else sigmoid
            if activation_fn == "sigmoid"
            else None
        )
        super(Hitat1, self).__init__(output_transform=output_transform, device=None)

    def reset(self):
        self.predictions_dict = {}
        self.actuals_dict = {}

    def update(self, output, masks=None):
        """
        Parameters
        ----------
        # predictions is n * num_class numpy array of predictions where n is number of samples
        # actuals is n * num_class numpy array of multihot labels where n is number of samples
        """
        predictions, actuals, ids = output

        for i, key in enumerate(ids):
            if key in self.predictions_dict:
                self.predictions_dict[key].append(predictions[i])
                self.actuals_dict[key].append(actuals[i])
            else:
                self.predictions_dict[key] = [predictions[i]]
                self.actuals_dict[key] = [actuals[i]]

    def compute(self):
        preds, acts, keys = [], [], []
        for key in self.predictions_dict.keys():
            pred_video = self.aggregation(
                np.stack(self.predictions_dict[key]), axis=0
            ).squeeze()
            if self.apply_activation:
                pred_video = self.apply_activation(pred_video)
            preds.append(pred_video)
            acts.append(np.stack(self.actuals_dict[key]).max(axis=0).squeeze())
            keys.append(key)
        preds = np.stack(preds)
        acts = np.stack(acts)

        non_negative = np.any(acts, axis=1)
        acts = acts[non_negative]
        preds = preds[non_negative]

        top_prediction = np.argmax(preds, 1)
        hits = acts[np.arange(acts.shape[0]), top_prediction]
        return np.average(hits)


class AveragePrecision(Metric):
    # compute metrics based on video ids
    def __init__(
        self,
        num_class=1000,
        top_n=None,
        aggregation="max",
        filter_empty_classes=False,
        output_transform=lambda x: x,
        activation_fn=None,
    ):
        super(AveragePrecision, self).__init__(
            output_transform=output_transform, device=None
        )
        self.num_class = num_class
        self.top_n = top_n
        self.filter_empty_classes = filter_empty_classes
        self.predictions = [[] for _ in range(num_class)]
        self.actuals = [[] for _ in range(num_class)]
        self.predictions_dict = {}
        self.actuals_dict = {}
        self.aggregation = np.max if aggregation == "max" else np.mean
        self.apply_activation = (
            softmax
            if activation_fn == "softmax"
            else sigmoid
            if activation_fn == "sigmoid"
            else None
        )

    def reset(self):
        self.predictions_dict = {}
        self.actuals_dict = {}

    def update(self, output):
        """
        Parameters
        ----------
        # predictions is n * num_class numpy array of predictions where n is number of samples
        # actuals is n * num_class numpy array of multihot labels where n is number of samples
        """
        predictions, actuals, ids = output

        for i, key in enumerate(ids):
            if key in self.predictions_dict:
                self.predictions_dict[key].append(predictions[i])
                self.actuals_dict[key].append(actuals[i])
            else:
                self.predictions_dict[key] = [predictions[i]]
                self.actuals_dict[key] = [actuals[i]]

    def compute(self):
        predictions, actuals, keys = self._arrange_predictions_by_class()

        res = []
        for i in range(self.num_class):
            target = np.concatenate(actuals[i])
            output = np.concatenate(predictions[i])
            ap_class, num_pos = self.ap(output, target, top_n=self.top_n)
            if not self.filter_empty_classes or num_pos > 0:
                res.append(ap_class)

        return res

    def _arrange_predictions_by_class(self):
        preds, acts, keys = [], [], []
        for key in self.predictions_dict.keys():
            pred_video = self.aggregation(
                np.stack(self.predictions_dict[key]), axis=0
            ).squeeze()
            if self.apply_activation:
                pred_video = self.apply_activation(pred_video)
            preds.append(pred_video)
            acts.append(np.stack(self.actuals_dict[key]).max(axis=0).squeeze())
            keys.append(key)
        preds = np.stack(preds)
        acts = np.stack(acts)

        predictions = [[] for _ in range(self.num_class)]
        actuals = [[] for _ in range(self.num_class)]
        for i in range(self.num_class):
            predictions[i].append(preds[:, i])
            actuals[i].append(acts[:, i])
        return predictions, actuals, keys

    @staticmethod
    def ap(predictions, actuals, top_n=None):
        num_positive_total = actuals.sum()
        if num_positive_total == 0:
            return float("NaN"), 0

        sorted_idx = np.argsort(predictions)[::-1]
        if top_n is not None:
            sorted_idx = sorted_idx[:top_n]
        actuals = actuals[sorted_idx]
        num_pos = actuals.sum()

        precisions = np.cumsum(actuals) / np.arange(1, len(actuals) + 1)
        ap = (precisions * actuals).sum() / (float(num_pos) + 1e-15)

        return ap, num_positive_total
