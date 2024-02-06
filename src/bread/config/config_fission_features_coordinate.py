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
        'x',
        'y',
    ],
    "edge_f_list": [
        "cmtocm_x",
        "cmtocm_y",
        "cmtocm_len",
        "cmtocm_angle",
        "contour_dist",
    ],
    "scale_time": 1,
    "nn_threshold": 36,
    "output_folder": "/mlodata1/hokarami/fari/tracking/generated_data/fissions/cell_graphs_features_locality",
    # "input_segmentations":[
    #     '/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SW182_01_segmentation.h5',
    #     '/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SW182_02_segmentation.h5',
    #     '/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_01_segmentation.h5',
    #     '/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_03_segmentation.h5',
    #     '/mlodata1/hokarami/fari/bread/data/fission/new_masks/240102_30C_fig_SX387_05_segmentation.h5',
    # ]
}
ass_graph_config = {
    "cellgraph_dirs": [
        # '/mlodata1/hokarami/fari/tracking/generated_data/fissions/cell_graphs_features_locality/240102_30C_fig_SW182_01_segmentation',
        # '/mlodata1/hokarami/fari/tracking/generated_data/fissions/cell_graphs_features_locality/240102_30C_fig_SW182_02_segmentation',
        # '/mlodata1/hokarami/fari/tracking/generated_data/fissions/cell_graphs_features_locality/240102_30C_fig_SX387_01_segmentation',
        # '/mlodata1/hokarami/fari/tracking/generated_data/fissions/cell_graphs_features_locality/240102_30C_fig_SX387_03_segmentation',
        # '/mlodata1/hokarami/fari/tracking/generated_data/fissions/cell_graphs_features_locality/240102_30C_fig_SX387_05_segmentation',
        "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features/wt_pom1D_01_07_R3D_REF_dv_trk_segmentation",
        "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features/wt_pom1D_01_15_R3D_REF_dv_trk_segmentation",
        "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features/wt_pom1D_01_20_R3D_REF_dv_trk_segmentation",
        "/mlodata1/hokarami/fari/tracking/generated_data/fissions/wt_cellgraphs_features/wt_pom1D_01_30_R3D_REF_dv_trk_segmentation",
        
    ],
    "output_folder": "/mlodata1/hokarami/fari/tracking/generated_data/fissions/ass_graphs_features_locality",
    "framediff_min": 1,
    "framediff_max": 4,
    "t1_max": -1, # default on -1
    "t1_min": 0
}

train_config = {
    "ass_graphs": "/mlodata1/hokarami/fari/tracking/generated_data/fissions/ass_graphs_features_locality",
    "result_dir": "/mlodata1/hokarami/fari/tracking/results/fissions/results_features",
    "dataset": "fission_train",
    "valid_dataset": '240102_30C_fig_SX387_05',
    "min_file_kb": 100,
    "filter_file_mb": 30, # filter training file sizes exceeding `filter_file_mb` MiB
    "dropout_rate": 0.01, # chekced
    "encoder_hidden_channels": 120,
    "encoder_num_layers": 4,
    "conv_hidden_channels": 120,
    "conv_num_layers": 3, # best was 5 (between 5 and 3) 
    "max_epochs" : 30,
    "lr": 0.001, 
    "weight_decay": 0.01, # (L2 regularization)
    "step_size": 1024, # steplr step size in number of batches (best is 512, at least for such data)
    "gamma": 0.5, # steplr gamma
    "cv": 1,
    "patience": 5,
    "scoring": 'valid_f1_ass'
}

test_config = {
    "ass_graphs": "/mlodata1/hokarami/fari/tracking/generated_data/fissions/ass_graphs_features_locality",
    "result_dir": "/mlodata1/hokarami/fari/tracking/results/fissions/results_features",
    "dataset": "fission_test",
    "filter_file_mb": 30, # filter file sizes exceeding `filter_file_mb` MiB
    'assignment_method': 'percentage_hungarian'

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