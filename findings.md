# Findings & Conclusion: Kitui Crop Yield Prediction

This document summarizes the key findings from the application of machine learning models to predict crop yields in Kitui County, Kenya, based on environmental and agricultural data from 2000–2024.

# Feature Importance  
![Feature Importance](./images/feature_importance.png)
- Rainfall emerged as the most critical variable affecting crop yields.
- Soil moisture also showed a significant positive correlation with output.
- Fertilizer use had weaker but consistent contributions to yield variation.

→ *These findings reinforce the need for effective water management and soil conservation practices.*
# Summary of Results

| Model                 | R² Score | MAE (tons/ha)  | RMSE (tons/ha) |
|-----------------------|----------|----------------|----------------|
| Random Forest (Tuned) | 0.928    | 0.210          | 0.305          |
| XGBoost (Tuned)       | 0.9765   | 0.1411         | 0.1957         |
| RNN (Tuned)           | 0.7882   | 0.4733         | 0.5879         |

- XGBoost delivered the most accurate yield predictions.
- Random Forest showed strong responsiveness to variable yield patterns.
- RNN produced smoother trends but struggled to capture sharp fluctuations.
- This plot shows how closely the model predictions match real crop yields. Points near the diagonal line indicate strong performance.
![Actual vs Predicted](./images/Actual_vs_Predicted_All_Models.png)


### Crop Performance Trends  
- Sorghum and pigeon peas exhibited higher resilience to climate variability than maize or beans.
- Yields of these drought-tolerant crops were more stable across years, especially during dry spells.
  
→ *This supports the adoption of drought-resilient crops in ASAL (arid and semi-arid land) regions like Kitui.*

### Hyperparameter Tuning Benefits  
- Using GridSearchCV, models were fine-tuned to improve learning rate, depth, and regularization parameters.
- This led to significant error reductions in both XGBoost and Random Forest, bringing predicted yields closer to observed values.

## Residual Analysis

- XGBoost and Random Forest models tracked actual yields more closely and adapted to sharp yield changes.
- RNN models produced smoother, less dynamic predictions, often staying near a fixed average (~5 tons/ha).
- Residual plots reveal prediction errors. Tighter clustering near zero means better model accuracy.
![Residuals](./images/Residuals_Distribution_All_Models.png)


# Recommendations

1. Prioritize Rainfall Monitoring & Forecasting: Rainfall remains the dominant predictor — improved early warning systems could directly enhance planting decisions.
2. Support Climate-Resilient Crops: Local governments should promote sorghum and pigeon peas, which consistently outperformed under variable conditions.
3. Adopt ML-Based Planning Tools: Stakeholders should integrate models like XGBoost into county-level agricultural dashboards to support targeted interventions.


# Conclusion

This study confirms that advanced machine learning models — particularly XGBoost — can significantly improve crop yield forecasting in drought-prone regions like Kitui County. By leveraging historical environmental data, these models deliver high predictive accuracy, enable better planning, and help mitigate risks posed by climate unpredictability.


For implementation details, model code, and dataset information, see the main [README.md](./README.md).

