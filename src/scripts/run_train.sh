#!/bin/bash 

# Set your project directory path here (YeaZ-toolbox)
PROJECT_DIR="/Users/farzanehwork/Documents/LPBS/codes/YeaZ-toolbox"

# Set the path for your input mask files should be one h5 file per movie.
DATA_PATH="/Users/farzanehwork/Desktop/LPBS_track_and_trace/Data_budding_yeast/"

# Activate conda environment
# source activate yeazNoTF &&
conda activate yeazNoTF #change the conda env to your own conda env

# navigate to the project directory
cd "$PROJECT_DIR" || { echo "Error: Could not navigate to $PROJECT_DIR/YeaZ-toolbox"; exit 1; }

# install the required packages, you can skip this step if you have already installed them.
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html && 
pip install torch_geometric && 
pip install -r requirements.txt && 
pip install -e .

# navigate to the scripts directory
cd "$PROJECT_DIR/src/scripts" || { echo "Error: Could not navigate to $PROJECT_DIR/src/scripts"; exit 1; }

# Preparing the data for training or testing.

# create cellgraphs. define the input segmentation file and the output folder. the output folder is used in the next step to create assgraphs.
# you can do this for multiple files by changing the input segmentation file. 
# you should have 2 different output folders for test and train sets. in each, you can have multiple movies.
# you can also specify the input segmentation and output folder in the config file to be able to run multiple files at once. (look at the sample config)

INPUT_SEGMENTATION="/Users/farzanehwork/Desktop/LPBS_track_and_trace/Data_budding_yeast/Recording_1_aka_Colony_1/colony001_segmentation.h5" # path to the input segmentation file
TRAIN_CELLGRAPH_DIR='/Users/farzanehwork/Desktop/LPBS_track_and_trace/generated/cellgraphs/train/' # path to the output folder
python -m bread.algo.tracking.build_cellgraphs --input_segmentation $INPUT_SEGMENTATION --output_folder $TRAIN_CELLGRAPH_DIR --config sample_config

# once for validation, and once for test set
INPUT_SEGMENTATION="/Users/farzanehwork/Desktop/LPBS_track_and_trace/Data_budding_yeast/Recording_2_aka_Colony_2/colony002_segmentation.h5" # path to the input segmentation file
VALID_CELLGRAPH_DIR='/Users/farzanehwork/Desktop/LPBS_track_and_trace/generated/cellgraphs/valid/' # path to the output folder
python -m bread.algo.tracking.build_cellgraphs --input_segmentation $INPUT_SEGMENTATION --output_folder $VALID_CELLGRAPH_DIR --config sample_config

INPUT_SEGMENTATION="/Users/farzanehwork/Desktop/LPBS_track_and_trace/Data_budding_yeast/Recording_3_aka_Colony_3/colony003_segmentation.h5" # path to the input segmentation file
TEST_CELLGRAPH_DIR='/Users/farzanehwork/Desktop/LPBS_track_and_trace/generated/cellgraphs/test/' # path to the output folder
python -m bread.algo.tracking.build_cellgraphs --input_segmentation $INPUT_SEGMENTATION --output_folder $TEST_CELLGRAPH_DIR --config sample_config

# create assgraphs, once for training, once for validation, and once for test set
TRAIN_ASSIGNMENT_GRAPH_DIR='/Users/farzanehwork/Desktop/LPBS_track_and_trace/generated/assgraphs/train' # path to the cellgraphs folder
python -m bread.algo.tracking.build_assgraphs --output_folder $TRAIN_ASSIGNMENT_GRAPH_DIR --cellgraphs_dir $TRAIN_CELLGRAPH_DIR --config sample_config 

VALID_ASSIGNMENT_GRAPH_DIR='/Users/farzanehwork/Desktop/LPBS_track_and_trace/generated/assgraphs/valid'  # path to the cellgraphs folder
python -m bread.algo.tracking.build_assgraphs --output_folder $VALID_ASSIGNMENT_GRAPH_DIR --cellgraphs_dir $VALID_CELLGRAPH_DIR --config sample_config 

TEST_ASSIGNMENT_GRAPH_DIR='/Users/farzanehwork/Desktop/LPBS_track_and_trace/generated/assgraphs/test'  # path to the cellgraphs folder
python -m bread.algo.tracking.build_assgraphs --output_folder $TEST_ASSIGNMENT_GRAPH_DIR --cellgraphs_dir $TEST_CELLGRAPH_DIR --config sample_config 

# train
TRAIN_SET=$TRAIN_ASSIGNMENT_GRAPH_DIR # path to the train set folder
VALID_SET=$VALID_ASSIGNMENT_GRAPH_DIR # path to the validation set folder
TEST_SET=$TEST_ASSIGNMENT_GRAPH_DIR # path to the test set folder
RESULTS_DIR=$RESULTS_DIR # path to the results folder
python train.py --train_set $TRAIN_SET --valid_set $VALID_SET --test_set $TEST_SET --config sample_config
