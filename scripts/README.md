# SPIRIT Execution Scripts

Four scripts to run the complete SPIRIT pipeline in sequence.

## Pipeline Execution

### 1. `run_data_processing.py`
Downloads NREL data and creates processed dataset.

```bash
python scripts/run_data_processing.py --config config/my_config.yaml
```


### 2. `run_embeddings.py`
Generates image embeddings using foundation models.

```bash
python scripts/run_embeddings.py --config config/my_config.yaml

# Resume from interruption
python scripts/run_embeddings.py --start-index 1500
```

### 3. `run_nowcasting.py`
Trains XGBoost model for current prediction.

```bash
python scripts/run_nowcasting.py --config config/my_config.yaml
```


### 4. `run_forecasting.py`
Trains Transformer model for multi-horizon prediction.

```bash
python scripts/run_forecasting.py --config config/my_config.yaml
```


## Complete Pipeline

```bash
cp config/config.yaml config/my_config.yaml
# Edit my_config.yaml with your settings

python scripts/run_data_processing.py --config config/my_config.yaml
python scripts/run_embeddings.py --config config/my_config.yaml
python scripts/run_nowcasting.py --config config/my_config.yaml
python scripts/run_forecasting.py --config config/my_config.yaml
```

## Common Issues

- **CUDA OOM:** Use `--batch-size 8`
- **HF Auth:** Run `huggingface-cli login`
- **Resume:** Use `--start-index` for embeddings
