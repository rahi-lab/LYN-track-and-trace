if __name__ == '__main__':
    from bread.data import Features, Segmentation
    import bread.algo.tracking as tracking
    from pathlib import Path
    from glob import glob
    from tqdm import tqdm
    import pickle, json
    import argparse
    import os, sys, datetime
    import torch
    from importlib import import_module

    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', dest='config', required=False, default='default', type=str, help='config file')
    parser.add_argument('--cellgraphs_dir', dest='cellgraphs_dir', required=False, default=None, type=str, help='filepaths to the cellgraph directories')
    parser.add_argument('--output_folder', dest='output_folder',required=False, default=None, type=str, help='output directory')
    parser.add_argument('--framediff_min', dest='framediff_min', required=False, default=None, type=int, help='minimum number of frame difference between two trackings')
    parser.add_argument('--framediff_max', dest='framediff_max', required=False, default=None, type=int, help='maximum number of frame difference between two trackings')
    parser.add_argument('--t1_max', dest='t1_max', type=int, required=False, default=None, help='maximum first frame')
    parser.add_argument('--t1_min', dest='t1_min', type=int, required=False, default=None, help='minimum first frame')

    args = parser.parse_args()
    config = import_module("bread.config." + args.config).configuration.get('ass_graph_config')
    if args.output_folder is not None:
        config['output_folder'] = args.output_folder
    if args.framediff_min is not None:
        config['framediff_min'] = args.framediff_min
    if args.framediff_max is not None:
        config['framediff_max'] = args.framediff_max
    if args.t1_min is not None:
        config['t1_min'] = args.t1_min
    if args.t1_max is not None:
        config['t1_max'] = args.t1_max
    if args.cellgraphs_dir is not None: 
        # set config["cellgraph_dirs"] to a list of every folder in args.cellgraphs_dir
        config["cellgraph_dirs"] = [os.path.join(args.cellgraphs_dir, folder) for folder in os.listdir(args.cellgraphs_dir) if os.path.isdir(os.path.join(args.cellgraphs_dir, folder))]        
    print("build assignment graph with args: ", config)

    
    print("build assignment graph with args: ", config)

    os.makedirs(Path(config["output_folder"]), exist_ok=True)
    extra = { 'node_attr': None, 'edge_attr': None, 'num_class_positive': 0, 'num_class_negative': 0 }
    node_attr = []
    edge_attr = []

    for cellgraph_dir in tqdm(config["cellgraph_dirs"], desc='cellgraph'):
        cellgraph_paths = list(sorted(glob(str(Path(cellgraph_dir) / 'cellgraph__*.pkl'))))
        cellgraphs = []
        for cellgraph_path in cellgraph_paths:
            with open(cellgraph_path, 'rb') as file:
                graph = pickle.load(file)
            cellgraphs.append(graph)

        if config["t1_max"] == -1:
            config["t1_max"] = len(cellgraphs)
        # check if "t1_min" is in the config dictionary
        if ("t1_min" in config):
            t1_min = config["t1_min"]
        else:
            t1_min = 0
            
        name = Path(cellgraph_dir).stem
        big_file_count = 0
        for t1 in tqdm(range(max(0,t1_min),min(len(cellgraphs), config["t1_max"])), desc='t1', leave=False):
            for t2 in tqdm(range(min(t1+config["framediff_min"], len(cellgraphs)), min(t1+config["framediff_max"]+1, len(cellgraphs))), desc='t2', leave=False):
                nxgraph = tracking.build_assgraph(cellgraphs[t1], cellgraphs[t2], include_target_feature=True)
                graph, node_attr, edge_attr = tracking.to_data(nxgraph, include_target_feature=True)
                save_path = Path(config["output_folder"]) / f'{name}__assgraph__dt_{t2-t1:03d}__{t1:03d}_to_{t2:03d}.pt'
                torch.save(graph, save_path)

                extra['num_class_positive'] += (graph.y == 1).sum()
                extra['num_class_negative'] += (graph.y == 0).sum()
                

    extra['node_attr'] = node_attr
    extra['edge_attr'] = edge_attr
    # convert a 0-dim tensor to a number
    extra['num_class_positive'] = int(extra['num_class_positive'])
    extra['num_class_negative'] = int(extra['num_class_negative'])

    with open(Path(config["output_folder"]) / 'metadata.txt', 'w') as file:
        file.write(f'Generated on {datetime.datetime.now()} with arguments {sys.argv}\n\n{config}')

    with open(Path(config["output_folder"]) / 'extra.json', 'w') as file:
        json.dump(extra, file)