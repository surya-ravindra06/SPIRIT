# SPIRIT: Short-term Prediction of solar IRradIance for zero-shot Transfer learning using Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2502.10307-b31b1b.svg)](https://arxiv.org/pdf/2502.10307)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

## Overview

SPIRIT is a novel approach leveraging foundation models for solar irradiance forecasting, enabling zero-shot transfer learning to new locations without historical data. Our method outperforms state-of-the-art models by up to **70%** in zero-shot scenarios.

## Repository Structure
```
SPIRIT/
├── README.md
├── requirements.txt
├── LICENSE
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── embeddings.py
│   ├── nowcasting.py
│   └── forecasting.py
├── scripts/
│   ├── __init__.py
│   ├── run_data_processing.py
│   ├── run_embeddings.py
│   ├── run_nowcasting.py
│   └── run_forecasting.py
└── utils/
    ├── __init__.py
    └── helper.py
```

### Key Features
- **Zero-shot Transfer Learning**: Deploy immediately at new solar sites without years of historical data collection.
- **Foundation Model Integration**: Utilizes pre-trained Vision Transformers (ViT) to extract rich sky image representations.
- **Physics-Informed Architecture**: Incorporates solar geometry, clear-sky modeling, and optimal panel orientation calculations
- **End-to-End Pipeline**: Automated data processing from raw satellite images to trained models.

### Real-Time Solar Irradiance Estimation (Nowcasting)
This model delivers immediate predictions of solar irradiance using a fusion of image-derived features and physics-informed signals.

- Input: A single sky image is processed through a Vision Transformer to extract high-level visual cues, which are combined with solar angles and clear-sky irradiance estimates.

- Model Design: An XGBoost regressor is trained on these inputs, using ensemble tree methods enhanced with depth control, subsampling strategies, and data-driven hyperparameter tuning to model complex, non-linear relationships.

- Target Output: Instantaneous prediction of Global Horizontal Irradiance (GHI) at the time of observation.

### Short-Term Solar Forecasting (1–4 Hours Ahead)
Designed for forward-looking energy planning, this model predicts solar irradiance trends based on recent temporal dynamics.

- Input: A rolling sequence of sky image embeddings over the past 6 hours, enriched with meteorological features and forward-projected clear-sky baselines.

- Architecture: A Transformer encoder maps the temporal structure of the input, using attention mechanisms and positional encoding. These outputs are passed through a residual MLP stack to model fine-grained temporal dependencies.

- Target Output: Forecasted GHI values across future 15-minute intervals, supporting horizons from 1 to 4 hours.



## Quick Start

### Installation
```bash
git clone https://github.com/your-username/SPIRIT.git
cd SPIRIT
pip install -r requirements.txt
```

### Configuration
```bash
# Copy and edit configuration
cp config/config.yaml config/my_config.yaml
# Edit with your paths, API keys, and parameters
```

### Usage
```bash
# 1. Process data
python scripts/run_data_processing.py --config config/my_config.yaml

# 2. Generate embeddings
python scripts/run_embeddings.py --config config/my_config.yaml

# 3. Train nowcasting model
python scripts/run_nowcasting.py --config config/my_config.yaml

# 4. Train forecasting model
python scripts/run_forecasting.py --config config/my_config.yaml
```

### Citation
If you use SPIRIT in your research, please cite our paper:
```bibtex
@article{spirit2025,
  title={SPIRIT: Short-term Prediction of solar IRradIance for zero-shot Transfer learning using Foundation Models},
  author={Aditya Mishra, T Ravindra, Srinivasan Iyengar, Shivkumar Kalyanaraman, Ponnurangam Kumaraguru},
  journal={arXiv preprint arXiv:2502.10307},
  year={2025}
}
```