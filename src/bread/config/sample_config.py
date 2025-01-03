import os
from importlib import import_module
from pathlib import Path

cell_graph_config = {
    "cell_f_list": [
        'area',
        'r_equiv',
        'r_maj',
        'r_min',
        'angel',
        'ecc',
        'maj_x',
        'maj_y',
        'min_x',
        'min_y',
    ],
    "edge_f_list": [
        "cmtocm_x",
        "cmtocm_y",
        "cmtocm_len",
        "cmtocm_angle",
        "contour_dist",
    ],
    "scale_time": 1,
    "nn_threshold": 12,
}

ass_graph_config = {
    "framediff_min": 1,
    "framediff_max": 1, # more if you want more distances between frames (unit: frame)
    "t1_max": 10, # default on -1 if you want all frames
    "t1_min": 0
}

train_config = {
    "min_file_kb": 0, # we used 100 because it worked better in training. 
    "filter_file_mb": 50, # filter training file sizes exceeding `filter_file_mb` MiB
    "dropout_rate": 0.01, # chekced
    "encoder_hidden_channels": 120,
    "encoder_num_layers": 4,
    "conv_hidden_channels": 120,
    "conv_num_layers": 3, # best was 5 (between 5 and 3) 
    "max_epochs" : 50,
    "lr": 0.001, 
    "weight_decay": 0.01, # (L2 regularization)
    "step_size": 1024, # steplr step size in number of batches (best is 512, at least for such data)
    "gamma": 0.5, # steplr gamma
    "cv": 1,
    "patience": 6,
    "scoring": 'valid_f1_ass'
}

test_config = {
    "filter_file_mb": 50, # filter file sizes exceeding `filter_file_mb` MiB
    "assignment_method": 'hungarian'
}
########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "cell_graph_config": {
        **cell_graph_config,
    },
    "ass_graph_config": {
        **ass_graph_config,
    },
    "train_config": {
        **train_config,
    },
    "test_config": {
        **test_config,
    },
    "use_wandb": True,
}