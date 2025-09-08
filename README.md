# Price Elasticity Model

This an implementation of that Alibaba paper from KDD 2021 about markdowns in e-commerce. I tweaked it a bit to use daily comparisons instead of their weekly approach because honestly daily gives you way better granularity.

## What it does

The model figures out how much demand changes when you discount products. Pretty straightforward - you feed it historical sales data with discounts, it learns elasticity patterns, then you can predict what happens if you run a 20% sale next week.

### The main tricks here

- Compares promo periods against the week before (when there was no promo)
- Gives more weight to recent data but you can tune how aggressive this is
- Each product category gets its own elasticity parameter 
- Longer promotions count more because they're more reliable data points
- The paired comparison naturally controls for product-specific stuff

## The math (if you care)

Basically we're modeling log-log relationships:
```
ln(promo_demand / baseline_demand) = elasticity × ln(discount_ratio) + intercept
```

The elasticity tells you the % change in demand for a 1% price change. So if elasticity is -2, a 10% discount roughly doubles your demand.

The loss function has temporal weighting and L2 regularization to avoid overfitting:
```
Loss = Σ weight × (actual - predicted)² + regularization × ||parameters||²
```

## Setup

```bash
# Just clone and install deps
git clone <repo>
cd price_elasticity
pip install numpy pandas scipy scikit-learn
```

## Quick examples

### Training

```python
from price_elasticity_model import PriceElasticityModel
import pandas as pd

# Load data, train, save
df = pd.read_csv('sales_data.csv')
model = PriceElasticityModel(forgetting_factor=0.995)  # 0.995 works well for stable products
metrics = model.fit(df)
print(f"R²: {metrics['weighted_r2']:.3f}")
model.save('models/elasticity_model.pkl')
```

### Using it for predictions

```python
model = PriceElasticityModel.load('models/elasticity_model.pkl')

# What happens if I run a 25% discount?
lift = model.predict_lift(discount_pct=0.25, category='Electronics')
print(f"25% off -> {lift:.2f}x demand")

# Check learned elasticities
elasticities = model.get_elasticities()
for cat, elast in elasticities.items():
    print(f"{cat}: {elast:.2f}")
```

### Real world usage

```python
# You have a base forecast from your team
base_forecast = 1000  
discount = 0.20  # planning a 20% sale

lift = model.predict_lift(discount, 'Electronics')
final = base_forecast * lift
print(f"Expected demand with 20% off: {final:.0f} units")
```

## What data you need

Your CSV needs these columns:
- `date`: YYYY-MM-DD format
- `sku`: product ID
- `category`: product category (Electronics, Clothing, etc)
- `discount_pct`: 0 to 1 (so 0.2 = 20% off)
- `demand`: actual units sold

**Minimum requirements:**
- 3+ months of daily data
- Each SKU needs at least 2 promo periods 
- Each category needs 3+ SKUs and 10+ total promos
- Make sure discount is 0 for regular price, >0 for sales

The cool thing is it shares learning across SKUs in the same category, so you don't need tons of data per individual product.

## Tuning tips

### Forgetting factor (how fast to forget old data)

- `0.90`: Fast fashion, trendy stuff (7-day half-life)
- `0.95`: Regular retail with some seasonality (14-day half-life)
- `0.99`: Stable grocery items (69-day half-life)  
- `0.995`: Super stable or when you have limited data (138-day half-life)

I usually start with 0.995 and adjust down if needed.

### Regularization (prevents overfitting)

- `0.5`: Noisy data, be conservative
- `0.1`: Standard choice, works for most
- `0.01`: Clean data, let it fit tighter

## Files

- `price_elasticity_model.py` - the actual model code
- `train_model.py` - script to train from command line
- `test_inference.py` - testing stuff
- `data/` - put your CSV files here
- `models/` - saved models go here

## Main functions

- `fit(df)` - trains on your data, returns metrics
- `predict_lift(discount, category)` - tells you the demand multiplier
- `get_elasticities()` - shows learned elasticities per category
- `save/load` - pickle the model

## Command line training

```bash
python train_model.py --data data/sales_data.csv --forgetting_factor 0.995
```

## Using in production

### Batch predictions

```python
# Load model once
model = PriceElasticityModel.load('models/elasticity_model.pkl')

# Apply to multiple products
for sku, category, discount in product_list:
    lift = model.predict_lift(discount, category)
    print(f"{sku}: {lift:.2f}x at {discount*100}% off")
```

## Common problems

**R² is too low:**
- Need more data or better quality data
- Try forgetting_factor=0.995 to use more history
- Lower the regularization

**Weird elasticities (positive or huge negative):**
- Bump up regularization to 0.5
- Check your discount data is correct
- Make sure you have good baseline periods

**Overfitting (great train, bad test):**
- More regularization
- Lower forgetting factor
- Check for data leakage

## Credits

Based on "Markdowns in E-Commerce Fresh Retail" from Alibaba (KDD 2021). I modified their approach to work better with daily data and added the tunable forgetting factor.
