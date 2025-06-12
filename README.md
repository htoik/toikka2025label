# Label Correction for Road Segmentation Using Roadside Cameras

TODO

## Dataset structure

The dataset should follow the following structure.

```
/path/to/dataset/
├── feeds/
│   ├── feed_name1/
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── 00000002.jpg
│   │   └── ...
│   └── feed_name2/
│       ├── 00000000.jpg
│       ├── 00000001.jpg
│       ├── 00000002.jpg
│       └── ...
└── masks/
    ├── feed_name1.jpg
    └── feed_name2.jpg
```

The dataset paths and frame indices should be setup in a YAML config, see [the example dataset structure](./data/example-dataset.yml).

## Installation

It is recommended to use conda for installation. You can install conda from [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

```bash
git clone https://github.com/htoik/toikka2025label
cd toikka2025label

conda env create -f environment.yml
conda activate toikka2025label
pip install -e .
```

## Usage

TODO

## Acknowledgment

TODO
