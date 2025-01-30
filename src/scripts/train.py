from typing import Optional
from bread.data import *
from bread.algo.tracking import AssignmentDataset, GNNTracker, AssignmentClassifier, GraphLoader, accuracy_assignment, f1_assignment

from glob import glob
from pathlib import Path
import datetime, json, argparse
import torch, os
from pprint import pprint
from sklearn.metrics import confusion_matrix


from skorch.dataset import ValidSplit
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint, ProgressBar, EpochScoring, WandbLogger, EarlyStopping
from skorch.callbacks import Callback


from trainutils import SaveHyperParams, seed

from importlib import import_module
import numpy as np
import pandas as pd




# os.environ['WANDB_API_KEY'] = WANDB_API_KEY
# os.environ['WANDB_CONSOLE'] = "off"
# os.environ['WANDB_JOB_TYPE'] = 'features_test'

def seed_torch(seed=42):
    # After all of this, still the seed seems not to be fixed!!
    import random
    import os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    
def test_pipeline(net, test_array, result_dir, data_name, assigment_method = None):
    results = {}
    results['gcn'] = {
        't1': [],
        't2': [],
        'colony':[],
        'confusion': [],
        'num_cells1': [],
        'num_cells2': [],
    }

    for file in test_array:
        # read graph from .pt file
        graph = torch.load(file)
        yhat = net.predict_assignment(graph, assignment_method=assigment_method).flatten()
        y = graph.y.squeeze().cpu().numpy()

        results['gcn']['confusion'].append(confusion_matrix(y, yhat))
        idt1_pattern = r'__(\d{3})_to'
        idt2_pattern = r'_to_(\d{3})'
        dt_pattern = r'_dt_(\d{3})'
        #  colony is anything before _segmentation 
        c_pattern = r'(\w+)_segmentation'
        idt1 = int(re.findall(idt1_pattern, file)[0])
        idt2 = int(re.findall(idt2_pattern, file)[0])
        dt = int(re.findall(dt_pattern, file)[0])
        colony = re.findall(c_pattern, file)[0]

        results['gcn']['t1'].append(idt1)
        results['gcn']['t2'].append(idt2)
        results['gcn']['colony'].append(colony)
        results['gcn']['num_cells1'].append(len(set([list(graph.cell_ids)[i][0] for i in range(len(graph.cell_ids))])))
        results['gcn']['num_cells2'].append(len(set([list(graph.cell_ids)[i][1] for i in range(len(graph.cell_ids))])))
    
    res = pd.DataFrame(results['gcn'])
    res['method'] = 'gcn'
    
    # num_cells = [len(seg.cell_ids(idt)) for idt in range(len(seg))]

    res['tp'] = res['confusion'].map(lambda c: c[1, 1])
    res['fp'] = res['confusion'].map(lambda c: c[0, 1])
    res['tn'] = res['confusion'].map(lambda c: c[0, 0])
    res['fn'] = res['confusion'].map(lambda c: c[1, 0])
    res['acc'] = (res['tp'] + res['tn']) / \
        (res['tp'] + res['fp'] + res['tn'] + res['fn'])
    res['f1'] = 2*res['tp'] / (2*res['tp'] + res['fp'] + res['fn'])
    # res['num_cells1'] = res['t1'].map(lambda t: num_cells[int(t)])
    # res['num_cells2'] = res['t2'].map(lambda t: num_cells[int(t)])
    res['timediff'] = 5 * (res['t2'] - res['t1'])
    res.drop(columns='confusion', inplace=True)

    # save results
    res.to_csv(result_dir/f'result_{data_name}_{assigment_method}.csv', index=False)
    res_sum = res.groupby(['timediff', 'colony'])[
    ['f1', 'tp', 'fp', 'fn', 'tn', 'num_cells1', 'num_cells2']].sum()
    res_sum['precision'] = res_sum['tp'] / (res_sum['tp'] + res_sum['fp'])
    res_sum['recall'] = res_sum['tp'] / (res_sum['tp'] + res_sum['fn'])
    res_sum['f1'] = 2*res_sum['tp'] / \
        (2*res_sum['tp'] + res_sum['fp'] + res_sum['fn'])
    res_sum[['f1', 'precision', 'recall']]
    print('result on test data: ', data_name)
    res_sum.to_csv(result_dir/f'result_sum_{data_name}_{assigment_method}.csv')
    print(res_sum)


class AssignmentCallback(Callback):
    def __init__(self):
        # self.classifier = classifier
        self.weight_sum = 0

    def on_epoch_end(self, net, **kwargs):
        for param in net.module_.parameters():
            self.weight_sum += param.sum().item()
        print(f'Sum of all weights: {self.weight_sum}')
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', dest='config', required=True, type=str, help='config file')
    parser.add_argument('--device', dest='device', default='cuda', type=str, help='device')
    parser.add_argument('--train_set', dest='train_set', default=None, type=str, help='directory to the assignment graphs of the training set')
    parser.add_argument('--valid_set', dest='valid_set', default=None, type=str, help='directory to the assignment graphs of the validation set')
    parser.add_argument('--test_set', dest='test_set', default=None, type=str, help='directory to the assignment graphs of the test set')
    parser.add_argument('--result_dir', dest='result_dir', default=None, type=str, help='directory to save the results')

    args = parser.parse_args()
    config = import_module("bread.config." + args.config).configuration
    
    pretty_config = json.dumps(config, indent=4)
    print(pretty_config)

    train_config = config.get('train_config')
    print("train with config: ", config)

    device = torch.device('cuda' if (torch.cuda.is_available() and args.device=='cuda') else 'cpu')
    
    if train_config['min_file_kb'] is not None:
        min_file_kb = train_config['min_file_kb']
        resultdir = Path(f'{args.result_dir}/_{str(min_file_kb)}KB_/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}')
    else:
        resultdir = Path(f'{args.result_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}')
    print("-- result directory --")
    print(resultdir)
    os.makedirs(resultdir, exist_ok=True)
    with open(resultdir / 'metadata.json', 'w') as file:
        json.dump(config, file)

    print('-- train arguments --')
    print(json.dumps(train_config, indent=4))

    # filter out files that are too large
    train_array = []
    test_array = []
    validation_array = []
    import re
    # prepare training dataset
    print('-- preparing datasets --')
    
    for filename in os.listdir(args.train_set):
        if filename.endswith(".pt"):
            train_array.append(os.path.join(args.train_set, filename))
    # filter file size larger than config['filter_file_mb'] for sake of training
    train_array = [ filepath for filepath in train_array if os.stat(filepath).st_size/2**20 < train_config["filter_file_mb"]]
    # filter file size smaller than config['min_file_kb'] for sake of training
    if train_config['min_file_kb'] is not None:
        train_array = [ filepath for filepath in train_array if os.stat(filepath).st_size/2**10 > train_config["min_file_kb"] ]

    # prepare validation dataset
    for filename in os.listdir(args.valid_set):
        if filename.endswith(".pt"):
            validation_array.append(os.path.join(args.valid_set, filename))
    # filter file size larger than config['filter_file_mb'] for sake of training
    validation_array = [ filepath for filepath in validation_array if os.stat(filepath).st_size/2**20 < train_config["filter_file_mb"]]
    # filter file size smaller than config['min_file_kb'] for sake of training
    if train_config['min_file_kb'] is not None:
        validation_array = [ filepath for filepath in validation_array if os.stat(filepath).st_size/2**10 > train_config["min_file_kb"] ]

    # prepare test dataset
    for filename in os.listdir(args.test_set):
        if filename.endswith(".pt"):
            test_array.append(os.path.join(args.test_set, filename))
    # filter file size larger than config['filter_file_mb'] for sake of training
    test_array = [ filepath for filepath in test_array if os.stat(filepath).st_size/2**20 < train_config["filter_file_mb"]]
    # filter file size smaller than config['min_file_kb'] for sake of training
    if train_config['min_file_kb'] is not None:
        test_array = [ filepath for filepath in test_array if os.stat(filepath).st_size/2**10 > train_config["min_file_kb"] ]

    train_dataset = AssignmentDataset(train_array)
    valid_dataset = AssignmentDataset(validation_array)
    test_dataset = AssignmentDataset(test_array)
    
    print('-- training dataset --')
    print(train_dataset)
    seed_value = 42
    seed_torch(seed=seed_value) # although we use seeding, but the results seems to be slightly different every time we train. 
 
    train_dataset = train_dataset.shuffle()

    # Create a wandb Run
    # wandb_run = wandb.init(config=config, group=args.config, project="GCN_tracker", reinit=True)

    # scoring_system = f1_assignment if train_config['scoring'] == 'valid_f1_ass' else accuracy_assignment
    # scoring_system = f1_assignment

    net = AssignmentClassifier(
        GNNTracker,
        module__num_node_attr=len(train_dataset.node_attr),
        module__num_edge_attr=len(train_dataset.edge_attr),
        module__dropout_rate=train_config["dropout_rate"],
        module__encoder_hidden_channels=train_config["encoder_hidden_channels"],
        module__encoder_num_layers=train_config["encoder_num_layers"],
        module__conv_hidden_channels=train_config["conv_hidden_channels"],
        module__conv_num_layers=train_config["conv_num_layers"],
        module__num_classes=1,  # fixed, we do binary classification
        max_epochs=train_config["max_epochs"],
        device=device,
        criterion=torch.nn.BCEWithLogitsLoss(
            # attribute more weight to the y == 1 samples, because they are more rare
            pos_weight=torch.tensor(100)
        ),
        optimizer=torch.optim.Adam,
        optimizer__lr=train_config["lr"],
        optimizer__weight_decay=train_config["weight_decay"],  # L2 regularization
        iterator_train=GraphLoader,
        iterator_valid=GraphLoader,
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False,
        batch_size=1,
        classes=[0, 1],
        # define valid dataset
        train_split=predefined_split(valid_dataset),
        callbacks=[
            LRScheduler(policy='StepLR', step_every='batch', step_size=train_config["step_size"], gamma=train_config["gamma"]),
            Checkpoint(monitor='valid_loss_best', dirname=resultdir, f_pickle='pickle.pkl'),
            SaveHyperParams(dirname=resultdir),
            EarlyStopping(patience=train_config["patience"]),
            ProgressBar(detect_notebook=False),
            # WandbLogger(wandb_run, save_model=True),
            EpochScoring(scoring=f1_assignment, lower_is_better=False, name='valid_f1_ass'),
            # AssignmentCallback(),
            # Evaluator(GraphLoader(test_dataset), "test" , lower_is_better=False)
        ],
    )

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    torch.cuda.empty_cache()
    
    print('-- starting training --')
    net.fit(train_dataset, y=None)
    
    print('-- starting testing --')
    test_config = config.get('test_config')
    
    filtered_result = test_pipeline(net, test_array, resultdir, 'test', assigment_method = test_config['assignment_method'])
    
    torch.cuda.empty_cache()

    
