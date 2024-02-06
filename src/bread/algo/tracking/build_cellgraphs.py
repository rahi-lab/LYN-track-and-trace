if __name__ == '__main__':
	from bread.data import Features, Segmentation, SegmentationFile, Microscopy
	import bread.algo.tracking as tracking
	import argparse
	from pathlib import Path
	from glob import glob
	from tqdm import tqdm
	import os, sys
	import pickle
	import datetime
	from importlib import import_module

	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--config', dest='config', required=False, default='default', type=str, help='config file')
	parser.add_argument('--input_segmentation', dest='segmentation', required=False, default=None, type=str, help='Address of a segmentation file')
	parser.add_argument('--output_folder', dest= 'output_folder', required=False, default=None , type=str, help='Address of a folder to save cellgraphs')
	parser.add_argument('--nn_threshold', dest = 'nn_threshold', default = None, type=int, help='threshold for nearest neighbor')
	parser.add_argument('--scale_time', dest = 'scale_time', default = None, type=int, help='scale time by this number')
	parser.add_argument('--fov', dest = 'fov', default = 'FOV0', type=str, help='if your segmentation file has multiple FOV, detrmine which one correlated with the microscopy files of the same name.')
	parser.add_argument('--frame_max', dest = 'frame_max', default = -1, type=int, help='maximum number of frames to process')
	parser.add_argument('--frame_min', dest = 'frame_min', default = 0, type=int, help='minimum number of frames to process')
	
	args = parser.parse_args()
	config = import_module("bread.config." + args.config).configuration.get('cell_graph_config')

	edge_features = config['edge_f_list']
	cell_features = config['cell_f_list']
	if args.output_folder is not None:
		config['output_folder'] = args.output_folder
	if args.nn_threshold is not None:
		config['nn_threshold'] = args.nn_threshold
	if args.scale_time is not None:
		config['scale_time'] = args.scale_time
	
	
	file_array = []
	if args.segmentation is not None:
		config['input_segmentations'] = [args.segmentation]
		input_segmentations = config['input_segmentations']
	else:
		input_segmentations = [args.segmentation]

	print("build cell graph args: ", config)

	for filename in input_segmentations:
		if filename.endswith(".h5"):
			file_array.append(filename)
	input_segmentations = file_array

	for fp_segmentation in tqdm(input_segmentations, desc='segmentation'):
		seg = SegmentationFile.from_h5(fp_segmentation).get_segmentation(args.fov)
		try:
			microscopy_file = Path(fp_segmentation).parent.parent /'microscopies'/ Path(fp_segmentation).stem
			# replace the name segmentation with microscopy in microscopy_file
			microscopy_file = str(microscopy_file).replace('segmentation', 'microscopy') + '.tif'
			microscopy = Microscopy.from_tiff(microscopy_file)
		except:
			microscopy = None
		feat = Features(seg, nn_threshold=config['nn_threshold'], scale_time=config['scale_time'])
		name = Path(fp_segmentation).stem
		os.makedirs(Path(config["output_folder"]) / name, exist_ok=True)

		if args.frame_max==-1:
			frame_max = len(seg)
		else:
			frame_max = args.frame_max

		for time_id in tqdm(range(max(0,args.frame_min), min(len(seg),frame_max)), desc='frame', leave=False):
			graph = tracking.build_cellgraph(feat, time_id=time_id, 
			cell_features=cell_features,
			edge_features=edge_features
			)
			if(graph is None):
				continue
			with open(Path(config["output_folder"]) / name / f'cellgraph__{time_id:03d}.pkl', 'wb') as file:
				pickle.dump(graph, file)

	with open(Path(config["output_folder"]) / 'metadata.txt', 'w') as file:
		file.write(f'Generated on {datetime.datetime.now()} with arguments {args.config}\n\n{config}')