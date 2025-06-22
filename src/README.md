# SPIRIT Core Modules

Four core modules implementing the SPIRIT solar forecasting pipeline.

## Modules

### `data_processing.py` - SolarDataProcessor
Downloads and processes solar data from NREL into structured datasets.

```python
processor = SolarDataProcessor(config)
df = processor.process_complete_pipeline()
```

**Output:** Dataset with images, weather data, and physics features on Hugging Face Hub

### `embeddings.py` - EmbeddingGenerator
Extracts embeddings from sky images using Vision Transformers.

```python
generator = EmbeddingGenerator(config)
generator.process_dataset(dataset_name)
```

**Supported Models:** ViT-Base/Large/Huge, Resnet, etc

**Output:** JSON file with 768-1280 dimensional embeddings per image depending on the model chosen.

### `nowcasting.py` - NowcastingModel
XGBoost model for current solar irradiance prediction.

```python
model = NowcastingModel(config)
final_model, params = model.train_final_model(dataset, embeddings, save_path)
```

**Features:** Image embeddings + solar geometry + clear-sky calculations

**Output:** Current GHI prediction (W/m²)

### `forecasting.py` - ForecastingModel
Transformer model for 1-4 hour ahead prediction.

```python
model = ForecastingModel(config)
final_model, params = model.train_final_model(dataset, embeddings, save_path)
```

**Architecture:** Transformer encoder + residual MLPs

**Input:** 1-hour sequence of embeddings + future clear-sky values

**Output:** 24-step ahead GHI predictions - 4 hours.

## Quick Usage

```python
from src import SolarDataProcessor, EmbeddingGenerator, NowcastingModel, ForecastingModel

config = load_config('config/config.yaml')

# Complete pipeline
processor = SolarDataProcessor(config)
generator = EmbeddingGenerator(config)
nowcast = NowcastingModel(config)
forecast = ForecastingModel(config)
```

**Requirements:** All modules use `config/config.yaml` for configuration

**GPU Recommended:** For embeddings and forecasting modules