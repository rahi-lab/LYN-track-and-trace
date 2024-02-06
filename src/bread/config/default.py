import os
from importlib import import_module
from pathlib import Path

cell_graph_config = {
    "cell_f_list": [
        'fourier',
        10,
        False,
        [
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
        ]
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
    "output_folder": "/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/cell_graphs_fourier_10_f10_locality_False",
    "input_segmentations":[
        '/mlodata1/hokarami/fari/bread/data/segmentations/colony006_segmentation.h5',
    ]
}

ass_graph_config = {
    "cellgraph_dirs": [
        "/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/cell_graphs_fourier_10_f10_locality_False/colony006_segmentation/",
    ],
    "output_folder": "/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/ass_graphs_fourier_10_f10_locality_False",
    "framediff_min": 1,
    "framediff_max": 4,
    "t1_max": -1, # default on -1
    "t1_min": 0
}

train_config = {
    "ass_graphs": "/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/ass_graphs_fourier_10_f10_locality_False",
    "result_dir": "/mlodata1/hokarami/fari/tracking/results/scaled_images/results_fourier_10_and_f10_locality_False",
    "dataset": "colony_0123478_test_set_1234567_dt_1234_t_all",
    "valid_dataset": 'colony001',
    "min_file_kb": 100,
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
    "ass_graphs": "/mlodata1/hokarami/fari/tracking/generated_data/scaled_images/ass_graphs_fourier_10_f10_locality_False",
    "result_dir": "/mlodata1/hokarami/fari/tracking/results/scaled_images/results_fourier_10_and_f10_locality_False",
    "dataset": "colony_56__dt_1234__t_all",
    "filter_file_mb": 50, # filter file sizes exceeding `filter_file_mb` MiB
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