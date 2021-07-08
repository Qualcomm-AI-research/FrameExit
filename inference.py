# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import fire
import numpy as np
import torch
import torch.nn as nn
from dataset.dataset import get_dataloader
from model.adaptive_models import ConditionalFrameExitInferenceModel
from utils.config import Config
from utils.metrics import AveragePrecision, Hitat1
from utils.misc import map_output_transform
torch.manual_seed(2)


def show_help():
    print("""Dataset preparation: 
            In this repository, we provide the evaluation script for two datasets: ActivityNet1.3 and Mini-kinetics. 
            You will need to first download the videos and then extract the frames as jpeg files using ffmpeg. 
            We use the following parameters to extract the frames:
            
            For ActivityNet1.3:
            'format': 'image2', 'r': 1, 'qscale:v': 2, 'vf': 'scale=350:350:force_original_aspect_ratio=increase'
            
            For Mini-Kinetics:
            'format': 'image2', 'r': 5, 'qscale:v': 2, 'vf': 'scale=256:256:force_original_aspect_ratio=increase'
            
            The data for both datasets should be structured as follows:
            
            |--path_frame 
              |--video1_name
                |frame_name_0001.jpeg
                |frame_name_0002.jpeg
                 .
              |--video2_name
                |frame_name_0001.jpeg
                |frame_name_0002.jpeg
                 .
                 
            ====================================================================================================
            
            # Annotations:
            For the ActivityNet1.3 download and place the following files under <data/activitynet> directory:
            wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json
            wget https://raw.githubusercontent.com/antran89/ActivityNet/master/Crawler/classes.txt
            
            Next, Use the following script to extract the validation videos:
            
            > import json
            > data = json.load(open('data/activitynet/activity_net.v1-3.min.json'))['database']
            > with open('data/activitynet/split_val_v1.3.txt', 'w') as f:
            >     for key, val in data.items():
            >         if data[key]['subset'] == 'validation':
            >             f.write('v_%s.mp4\n' % key)
            
            -----------------------
            
            For Mini-kinetics download and place the following files under `<data/minikinetics>` directory:
            wget https://raw.githubusercontent.com/Alvin-Zeng/GCM/master/anet_toolkit/Crawler/Kinetics/data/kinetics_val.csv
            wget https://raw.githubusercontent.com/mengyuest/AdaFuse/master/data/kinetics/minik_classInd.txt
            wget https://raw.githubusercontent.com/mengyuest/AdaFuse/master/data/kinetics/mini_val_videofolder.txt
            
            Next, Use the following script to extract the validation videos:
            
            > lines = [line.rsplit('/')[-1].rsplit(' ')[0]+'.mp4' for line in open('data/minikinetics/mini_val_videofolder.txt')]
            > with open('data/minikinetics/split_val_minikinetics.txt', 'w') as f:
            >     for line in lines:
            >         f.write('%s\n' % line)
            
            ====================================================================================================
            """
          )


def inference(*config_paths, **kwargs):
    # load config file
    config = Config.load(*config_paths, **kwargs)

    if config.help:
        show_help()
        return

    num_class = config.model.num_class
    dataloader = get_dataloader(config.data)

    # load model
    model = ConditionalFrameExitInferenceModel(config)
    model = nn.DataParallel(model.cuda())
    state = torch.load(config.checkpoint.init)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded from: %s" % config.checkpoint.init)

    # define metrics
    metric = {
        "hit_1": Hitat1(
            output_transform=map_output_transform(video_level=True), aggregation="mean"
        ),
        "map": AveragePrecision(
            output_transform=map_output_transform(video_level=True), num_class=num_class
        ),
    }

    n_processed_video = 0
    exit_stats = []
    for itr, batch in enumerate(dataloader):
        x, (y, metadata) = batch
        frame_ids = metadata["frame_ids"][0]
        y = y.squeeze(dim=1)
        index = torch.tensor([5, 0, 9, 2, 7, 4, 6, 3, 8, 1], dtype=torch.long)
        x = torch.index_select(x, 1, index)
        x = x.to(device="cuda")
        model.eval()

        z_previous = None
        with torch.no_grad():
            for t in range(x.shape[1]):
                y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t)
                if y_pred is not None:  # exit if true
                    exit_stats.append(t)
                    break

            prob = nn.Softmax(dim=1)(y_pred).detach().cpu()

        # update metrics
        for k, m in metric.items():
            m.update((prob, y, frame_ids))

        n_processed_video += x.shape[0]

        print(f"test: iter:{itr}/{len(dataloader)}")

    print("number of processed videos: %d" % n_processed_video)
    for name, m in metric.items():
        print("%s: %f" % (name, np.nanmean(m.compute())))

    arch_name = config.model.backbone.name
    single_frame_mac = (
        4.12 if "resnet50" in arch_name else 1.8 if "efficientnet" in arch_name else 0
    )
    hist, bin_edges = np.histogram(np.array(exit_stats), bins=range(0, 11))
    print("Exiting statistics:")
    print(", ".join("  {:.0f}  ".format(f) for f in bin_edges[1:]))
    print(", ".join("{:.2f}%".format(f) for f in hist / hist.sum() * 100))
    print("Model Mac: ", np.mean(np.array(exit_stats) + 1) * single_frame_mac)


if __name__ == "__main__":
    fire.Fire(inference)
