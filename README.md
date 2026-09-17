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

## Results

The XGBoost model was able to recover the ecological patterns that were intentionally built into the simulated data.

### Temperature threshold

The SHAP temperature plot shows that temperature contributes positively to predicted biomass at lower temperatures, but its contribution becomes increasingly negative above approximately 30°C.

This reflects the thermal threshold that was built into the simulation.

### Soil moisture × nitrogen interaction

The SHAP interaction analysis showed that the strongest pairwise interaction was between soil moisture and soil nitrogen, with a mean absolute interaction value of approximately 0.45.

This is consistent with the simulated relationship where nitrogen contributes to biomass mainly when soil moisture is above 20%.

### Interpretation

These results demonstrate how an interpretable machine learning workflow can recover nonlinear relationships and interactions from environmental data.

However, these are not new ecological discoveries because the relationships were deliberately defined when the simulated data was created. With real ecological data, similar model patterns could be used to generate ecological hypotheses for further investigation.




