# Big Data-Driven Identification of Shared Bicycle Parking Aggregation Areas and Tidal Demand Patterns

This repository provides the official implementation of two clustering algorithms introduced in the paper:
"Big Data-Driven Identification of Shared Bicycle Parking Aggregation Areas and Tidal Demand Patterns"
by Hanqiang Qian, Jiachen Wang, Shuyan Zheng, Yue Shi, and Yanyan Chen (2025).

## 📖 Overview

Dockless shared bicycles are widely used in urban areas, but unregulated parking introduces management challenges. This project implements two complementary algorithms:

### AD-HDBSCAN (Adaptive Density HDBSCAN)

- An adaptive density-based clustering algorithm for identifying shared bicycle parking aggregation areas.
- Utilizes concentration index and MST-based connected components.
- Achieves +13.93% improvement compared to conventional algorithms.

### Contrastive Clustering for Time Series

- A deep learning–based contrastive learning model for high-dimensional temporal features.
- Handles multidimensional sequences (borrow, return, park) with data augmentation.
- Identifies distinct demand patterns with tidal characteristics.

Together, these approaches support managers in delineating virtual stations and designing demand-driven scheduling strategies.

## 📂 Project Structure

```
virtual-station-code/
├── ADHDBSCAN.py                     # Adaptive density clustering pipeline (Steps 1–4)
├── contrastive_clustering_train.py  # Contrastive clustering with data augmentation + training
├── deeplearning_models/             # Deep learning models
│   ├── Fc_model.py                  # Fully connected network
│   └── layers/                      # Neural network layers
├── modules/                         # Core modules (transform, resnet, loss functions)
├── utils/                           # Utility functions
├── data/                            # Example input data (CSV/shape files)
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

## ⚙️ Installation

```bash
git clone https://github.com/your-repo/virtual-station-code.git
cd virtual-station-code
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Adaptive HDBSCAN

Run the four-step pipeline:

```bash
python ADHDBSCAN.py
```

**Input:**
- `beijing_bike_positions.csv`
- `beijing_bike_positions_taz.csv`
- `gis_layer/ddc2006_84.shp`

**Output:**
- `clustering_output/HDBSCAN_adaptive_params.shp`
- `clustering_output/images/HDBSCAN_adaptive_params.png`

### 2. Contrastive Clustering

Train the contrastive clustering model on time series:

```bash
python contrastive_clustering_train.py
```

**Input:**
- `borrow_sequence.csv`
- `return_sequence.csv`
- `stop_sequence.csv`

**Output:**
- `BikeFeatureSequence_ContrastiveClustering.pkl`
- Saved models in `save/ContrastiveClustering_Fc_model*/`

## 📊 Results

- **AD-HDBSCAN** identifies parking aggregation areas across varied densities with improved precision.

- **Contrastive Clustering** extracts demand patterns:
  - Tidal peaks during commuting hours
  - Balanced usage near transport hubs
  - Stable or cumulative patterns in residential/office areas

## 📌 Citation

If you use this code, please cite:

```bibtex
@article{qian2025bikedemand,
  title={Big Data-Driven Identification of Shared Bicycle Parking Aggregation Areas and Tidal Demand Patterns},
  author={Qian, Hanqiang and Wang, Jiachen and Zheng, Shuyan and Shi, Yue and Chen, Yanyan},
  journal={Preprint submitted to Elsevier},
  year={2025}
}
``` 