# 1. import necessary libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# 2. Create the simulated environmental Data
np.random.seed(42)
n_samples = 1500

temperature = np.random.uniform(10, 38, n_samples)
soil_moisture = np.random.uniform(5, 45, n_samples)
soil_nitrogen = np.random.uniform(10,80, n_samples)
canopy_cover = np.random.uniform(0, 100, n_samples)

# 3. Create the ecological relationships

thermal_effect = np.where(temperature > 30, 
                          -2.5 * (temperature -30) ** 1.5,
                          0.4 * temperature
                          )

moisture_n_interaction = np.where(soil_moisture > 20, 
                                  0.08 * (soil_moisture - 20) * (soil_nitrogen / 10),
                                  0.0
                                  )

#This Creates the two ecological patterns we want the model to learn:
#- Temperature: biomass increases up to ~30°C, then drops sharply.
#- Moisture × nitrogen: nitrogen helps biomass only when moisture is above 20%."""

# 4. Calculate plant biomass
# This calculates aboveground plant biomass for each observation (Our target variable)

biomass = (40
           + thermal_effect
           +moisture_n_interaction
           +0.15 * canopy_cover
           + np.random.normal(0,4, n_samples))

X = pd.DataFrame({'temperature': temperature,
                  'soil_moisture': soil_moisture,
                  'soil_nitrogen': soil_nitrogen,
                  'canopy_cover': canopy_cover})

y= biomass

# 6. Train the XGBoost Model

# We choose XGBoost because our ecological relationships are expected to potentially be nonlinear,
# threshold like and interactive and tree based models are good at representiong
#those kinds of relationships

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

# 7. Calculate SHAP Values

#SHAP explains how each feature contributes to the model's predictions

explainer =shap.TreeExplainer(model) #Treeexplainer is specifially designed to explain tree based models
shap_values = explainer(X)

# Visualise the temperature effect
# This plot shows how temperature contributes to predicted
# biomass across the observations.
# We colour the points by canopy cover.

shap.plots.scatter(
    shap_values[:, "temperature"],
    color=shap_values[:, "canopy_cover"],
    show=False
)
plt.savefig("results/temperature_shap.png", bbox_inches="tight")
plt.close()

# 9. Visualise the soil moisture × nitrogen relationship

# This plot shows how soil moisture contributes to predicted
# biomass.
# Points are coloured by soil nitrogen to help examine the
# moisture × nitrogen relationship.

shap.plots.scatter(
    shap_values[:, "soil_moisture"],
    color=shap_values[:, "soil_nitrogen"],
    show=False
)
plt.savefig("results/moisture_nitrogen_shap.png", bbox_inches="tight")
plt.close()

# 10. Calculate SHAP interaction values
# shap_interaction_values = explainer.shap_interaction_values(X)
# print(shap_interaction_values.shape)

# We use the first 100 observations because interaction values
# are more computationally expensive than regular SHAP values.

shap_interaction_values = explainer.shap_interaction_values(X[:100])
print(shap_interaction_values.shape)

# 11. Quantify pairwise interactions
# Calculate the average absolute interaction strength for
# each pair of environmental variables.

feature_names = X.columns

mean_abs_interactions = np.abs(shap_interaction_values).mean(axis=0)

for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        print(
            feature_names[i],
            "×",
            feature_names[j],
            ":",
            mean_abs_interactions[i, j]
        )

# 12. Interpret the results        
# The result tells us that the strongest interaction is         
# soil moisture × soil nitrogen = 0.450

# This is consistent with the relationship intentionally
# built into the simulated data.
#
# The temperature SHAP plot shows the model's response changing
# around the ~30°C threshold.
#
# In this controlled simulation, the XGBoost model and SHAP
# analysis recovered the threshold and interaction patterns
# that were intentionally embedded in the data.
#
# These are not new ecological discoveries because the
# relationships were defined when the simulated data was created.
# In real ecological data, similar model patterns could be used
# to formulate ecological hypotheses for further investigation.