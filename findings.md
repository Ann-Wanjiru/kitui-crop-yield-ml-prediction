# Findings & Conclusion: Kitui Crop Yield Prediction

This document summarizes the key findings from the application of machine learning models to predict crop yields in Kitui County, Kenya, based on environmental and agricultural data from 2000–2024.

# Summary of Results

| Model                 | R² Score | MAE (tons/ha)  | RMSE (tons/ha) |
|-----------------------|----------|----------------|----------------|
| Random Forest (Tuned) | 0.928    | 0.210          | 0.305          |
| XGBoost (Tuned)       | 0.9765   | 0.1411         | 0.1957         |
| RNN (Tuned)           | 0.7882   | 0.4733         | 0.5879         |

- XGBoost delivered the most accurate yield predictions.
- Random Forest showed strong responsiveness to variable yield patterns.
- RNN produced smoother trends but struggled to capture sharp fluctuations.


## Key Insights

- Rainfall was identified as the most impactful variable in determining crop yield.
- Soil moisture also significantly influenced yield levels.
- Drought-resilient crops such as sorghum and pigeon peas outperformed others in consistency.
- Hyperparameter tuning using Grid Search greatly improved model accuracy.
- Machine learning models, especially tree-based ones, successfully captured complex nonlinear patterns in the data.

## Conclusion

This project illustrates how integrating historical agro-environmental data with machine learning models like XGBoost and Random Forest can significantly enhance the accuracy, interpretability, and applicability of crop yield predictions.

These insights can directly support:
- Farmers with better planting and irrigation planning  
- Policymakers in Kitui County with resource allocation  
- Researchers aiming to expand AI use in agriculture  
- Stakeholders promoting food security and sustainability  

For implementation details, model code, and dataset information, see the main [README.md](./README.md).

