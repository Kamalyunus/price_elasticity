# Parametric Price Elasticity Model

Implementation of the **parametric price elasticity component** from the KDD 2021 paper "Markdowns in E-Commerce Fresh Retail: A Counterfactual Prediction and Multi-Period Optimization Approach" by Alibaba Group.

## Overview

This implementation provides:
- **Offline training** on historical sales data
- **Production inference** for real-time price optimization
- **Hierarchical normalization** for improved stability
- **Clean integration** with existing forecasting systems

## Mathematical Foundation

**Training equation with normalization:**
```
ln(Y_i / Y_norm) = (θ_1 + θ_2^T L_i) × ln(d_i) + c
```

**Inference equation:**
```
ln Y_i,t(d_i) = θ^T L̂_i (ln d_i - ln d_i,t^o) + ln Y_i,t^o
```

Where:
- `Y_i`: Demand for product i
- `Y_norm`: Normalization constant (category average recommended)
- `d_i`: Discount ratio (1 - discount_percentage)
- `θ^T L̂_i`: Category-specific elasticity parameter

## Key Features

✅ **Category-level normalization** - More stable than product-level  
✅ **Hierarchical elasticity** - Category-specific parameters  
✅ **Works with any forecast** - Compatible with existing systems  
✅ **Fast inference** - Simple exponential calculation  
✅ **Production ready** - Clean offline training + inference  

## Installation

```bash
# Create and activate virtual environment
python3 -m venv price
source price/bin/activate  # On Windows: price\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Quick Start

### Run with Sample Data

```bash
# 1. Activate virtual environment
source price/bin/activate

# 2. Train model on sample data
python train_elasticity_offline.py \
    --data data/generated_data.csv \
    --output models/elasticity_model.pkl

# 3. Test the trained model
python test_inference.py
```

## Usage

### 1. Training (Offline)

Train on your historical sales data:

```bash
# Activate environment
source price/bin/activate

# Train with recommended settings
python train_elasticity_offline.py \
    --data your_sales_data.csv \
    --output models/elasticity_model.pkl \
    --normalization category_mean \
    --regularization 0.1
```

**Required data columns:**
- `date`: Transaction date
- `sku`: Product identifier
- `category`: Product category
- `base_price`: Original price
- `actual_price`: Price after discount
- `demand`: Sales quantity

**Normalization options:**
- `category_mean`: Category averages ⭐ **Recommended**
- `product_mean`: Individual product averages (may be unstable)
- `global_mean`: Global average (most stable)

### 2. Inference (Production)

Apply elasticity to base forecasts:

```bash
# Run inference examples
source price/bin/activate
python elasticity_inference.py
```

Or use in your Python code:

```python
from elasticity_inference import ElasticityInference

# Load trained model
inference = ElasticityInference('models/elasticity_model.pkl')

# Your forecasting team's base forecast (at base price)
base_forecast = 1000

# Get final forecast at 20% discount
final_forecast = inference.predict(
    base_forecast=base_forecast,
    discount_pct=0.2,
    category='cat_0'  # Use your actual category name
)

print(f"Final forecast: {final_forecast:.0f} units")
```

### 3. Batch Processing

```python
import pandas as pd

forecasts = pd.DataFrame({
    'sku': ['PROD_001', 'PROD_002'],
    'category': ['Electronics', 'Groceries'],
    'base_forecast': [1000, 2000],
    'discount_pct': [0.1, 0.25]
})

results = inference.predict_batch(forecasts)
print(results[['sku', 'base_forecast', 'final_forecast']])
```

## Normalization Strategy

Following the paper's recommendation for stability:

| Method | Description | Stability | Use Case |
|--------|-------------|-----------|----------|
| `category_mean` | Average sales of product's category | ⭐ High | **Recommended default** |
| `product_mean` | Average sales of individual product | ⚠️ Low | Products with rich history |
| `global_mean` | Average sales across all products | ✅ Highest | Maximum stability |

**From the paper:** *"The normalized factor Y_norm_i is the average sales of product i by definition. However, this quantity may be not very stable. In practice, the average sales of level-2 or level-3 category that the product belongs to can be used for normalization."*

## Integration with External Forecasts

The model seamlessly works with forecasts from your existing forecasting team:

### **Key Insight:**
The normalization constant cancels out during inference, so the model works with **any base forecast** (normalized or not):

```python
Final Demand = Base Forecast × exp[θ × (ln(d_target) - ln(d_base))]
```

### **Workflow:**
1. **Your team provides**: Base demand forecast at base price (no discount)
2. **Elasticity model adds**: Price sensitivity adjustment using learned θ
3. **Output**: Final demand forecast at target discount level

## Configuration

Configure via `config.yaml`:

```yaml
model:
  regularization_strength: 0.1
  use_normalization: true
  normalization_method: "category_mean"  # Recommended

training:
  test_split: 0.2

evaluation:
  results_dir: "models/"
  save_model: true
```

## Project Structure

```
price_elasticity/
├── parametric_elasticity_model.py  # Core elasticity model implementation
├── train_elasticity_offline.py     # Training script
├── elasticity_inference.py         # Production inference interface
├── test_inference.py               # Test script with examples
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── data/
│   └── generated_data.csv         # Sample data for testing
├── models/                         # Trained models directory
│   ├── elasticity_model.pkl       # Trained model (after training)
│   └── elasticity_model.json      # Model metadata
└── price/                          # Virtual environment (created during setup)
```

## Elasticity Interpretation

| Elasticity | Interpretation | 10% Discount Response |
|------------|----------------|----------------------|
| -0.5 | Inelastic | +5% demand |
| -1.0 | Unit elastic | +10% demand |
| -1.5 | Moderately elastic | +15% demand |
| -2.0 | Elastic | +20% demand |
| -2.5+ | Highly elastic | +25%+ demand |

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{zhao2021markdowns,
  title={Markdowns in E-Commerce Fresh Retail: A Counterfactual Prediction and Multi-Period Optimization Approach},
  author={Zhao, Yuyang and others},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
  pages={3343--3353},
  year={2021}
}
```