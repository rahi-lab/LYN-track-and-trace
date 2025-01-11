# ``LYN-track-and-trace`` -- A package to analyze yeast microscopy data

This package is designed to analyze microscopy data of yeast cells. It provide algorithms for tracking and lineage tracing of yeast cells. It also provides a graphical user interface to visualize the data, and the performing various lineage tracing algorithms, editing, and extracting lineage tracing results.
The tracking algorithms are used in the YeaZ application located at https://github.com/rahi-lab/YeaZ-GUI. Using this GUI application, you can use our tracking algorithms to track yeast cells in your own data.

## Installation

Recommended : create a virtual environment :

```sh
conda create -n LYN-track-and-trace python=3.9
conda activate LYN-track-and-trace
python -m pip install pip --upgrade
```

Installation of the package:

```sh
# Install the package in development mode (installs dependencies and creates symlink)
git clone https://github.com/rahi-lab/LYN-track-and-trace
# alternatively, download and extract the zip of the repository
cd LYN-track-and-trace
pip install -r requirements.txt
pip install -e .
```

## Graphical user interface for lineage tracing

Launch the GUI using ``lyn-trace-gui``.

### visualize microscopy files

To load microscopy data, click on "add nd2 or tif files". you can load a single nd2 file containing multiple timestamps and one or multiple channels, or you can load multiple tif files, each containing a single channel with multiple timestamps. The GUI will automatically detect the number of channels and timestamps in the data. You can then select the channel and timestamp you want to visualize.

To load segmentation masks, you can use the open segmentation button which will open a file dialog to select the segmentation masks. The segmentation masks should be an h5 file which can have multiple fields of view. The segmentation masks should be binary images where each cell is represented by a different label.

All the different channels of microscopy movie and the mask will appear overlaying each other. you can control the transparency of the each layer using the slider on the right control panel. 
The bottom control panel give you the ability to change the field of view to show, and wether to show the cell ids and lineage tracing (if exists). You can also navigate through the timestamps using the buttons on this control panel.

### lineage tracing
you have the option to open an existing lineage tracing file using file->open lineage tracing button (on the top bar). The lineage tracing file should be a csv file with the following columns: "bud_id", "parent_id", and "time".
To run lineage tracing algorithm, you have 2 opetions:

1- Lineage tracing using a budneck marker. This algorithm will use the peak of the budneck marker to determine the parent cell of each bud. To do so, you need to choose guess lineage using budneck under the new on the top menu. 

2- Lineage tracing using a neural network. This algorithm is recommended when a budneck marker is not available and you want to trace lineage only using the mask of cells. This algorithm will use a neural network to determine the parent cell of each bud based on position and distance of cell from the bud. To do so, you need to choose guess lineage using NN under the new on the top menu.

Both algorithms have a series of parameters that you can adjust to get the best results. They will output a table with the lineage tracing results and the confidence score for each lineage tracing. By clicking on each cell in the lineage tracing table, the visualization will navigate to corresponding cell and you can manually check the tracing and correct it in the table if needed.
In special cases parent ids in this table might be negative number which means that the algorithm cannot determine the parent of this cell for the following reasons:

- the cell is in the first frame of the movie therefor the parent can't be determined (parent is -1)
- the algorithm couldn't find any mother for this cell or the parent of cell is out of the field of view(parent is -2)
- the algorithm needs more frames to determine the parent of this cell (parent is -3)

### save lineage tracing

To save the lineage tracing results, you can use the save lineage option under file menu on the top menu. This will save the lineage tracing results in a csv file with the same format as the input lineage tracing file.

