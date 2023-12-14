# ``yeaz-toolbox`` -- A package to analyze YeaZ data (or yeast data in general)

## Installation

Recommended : create a virtual environment :

```sh
conda create -n yeaz_toolbox python=3.9
conda activate yeaz_toolbox
python -m pip install pip --upgrade
```

Installation of the package:

```sh
# Install the package in development mode (installs dependencies and creates symlink)
git clone https://github.com/rahi-lab/YeaZ-toolbox
# alternatively, download and extract the zip of the repository
cd YeaZ-toolbox
pip install -r requirements.txt
pip install -e .
```

## Graphical user interface

Launch the GUI using ``python -m bread.gui``.

