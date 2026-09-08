# Interpretable ML for Ecological Threshold and Interaction Discovery
This project shows how machine learning can help identify patterns in ecological data.
We use XGBoost to predict plant biomass and SHAP to understand what the model has learned.

## Overview
Ecological relationships are not always simple. For example:
- Plants may grow well up to a certain temperature, then growth may drop quickly.
- Nitrogen may only help plants when there is enough soil moisture.
- Different environmental factors can work together to affect plant growth.

This project uses simulated data to demonstrate how machine learning and SHAP can help find these patterns.

## The Workflow
1. Create simulated ecological data.
2. Train an XGBoost model to predict plant biomass.
3. Use SHAP to see how each environmental factor affects the prediction.
4. Create plots to identify non-linear relationships and interactions.
5. Calculate SHAP interaction values between environmental variables.

### The Dataset
The dataset contains four environmental variables:

**Temperature (10–38°C)**: Plant biomass starts to decrease sharply when temperature goes above 30°C.
**Soil Moisture (5–45%)**: Higher moisture can increase plant growth.
**Soil Nitrogen (10–80 mg/kg)**: Nitrogen increases growth mainly when soil moisture is above 20%.
**Canopy Cover (0–100%)**: More canopy cover has a positive effect on biomass.


## Requirements
Python




