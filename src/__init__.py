"""
SPIRIT: Short-term Prediction of solar IRradIance for zero-shot Transfer learning using Foundation Models

This package contains the core modules for the SPIRIT solar irradiance forecasting system.
"""

from .data_processing import SolarDataProcessor
from .embeddings import EmbeddingGenerator
from .nowcasting import NowcastingModel
from .forecasting import ForecastingModel

__version__ = "1.0.0"
__author__ = "SPIRIT Team"
__email__ = "suryaravindra01@gmail.com"

__all__ = [
    "SolarDataProcessor",
    "EmbeddingGenerator", 
    "NowcastingModel",
    "ForecastingModel"
]