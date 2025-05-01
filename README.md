# kitui-crop-yield-ml-prediction
Machine learning–based crop yield prediction for Kitui County using XGBoost, Random Forest, and RNN.

## Overview  
This repository provides synthetic data and Python code for predicting crop yields in Kitui County, showcasing advanced workflows in data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents  
- [Technologies Used](#technologies-used)  
- [Data](#data)  
- [Feature Engineering](#feature-engineering)  
- [Modeling Techniques](#modeling-techniques)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [License](#license)  

## Technologies Used  
- **Python**  
- **pandas** & **NumPy** for data manipulation  
- **scikit-learn** (Random Forest)  
- **XGBoost** for gradient boosting models  
- **TensorFlow** / **Keras** for RNN implementation  
- **Matplotlib** & **Seaborn** for visualization  

## Data  
- *Kitui_Weekly.csv* (located in the Data/ folder):
A weekly dataset containing historical crop yield information along with key environmental variables such as rainfall (mm), soil moisture (%), temperature (°C), and fertilizer usage (kg/ha) for various crops in Kitui County from 2000 to 2024.

## Feature Engineering  
 (located in the Scripts/ folder)
- **Time-Series Upsampling:** Converted annual crop-yield records into a weekly frequency by use of calibrated random noise derived from the original distribution, preserving key seasonal trends while enabling finer-grained modelling.

## Modeling Techniques  
(located in the Scripts/ folder)
1. **XGBoost**  
   - Optimized gradient boosting for high accuracy and performance.  
2. **Random Forest**  
   - Ensemble of decision trees for robust, interpretable results.  
3. **Recurrent Neural Network (RNN)**  
   - Captures temporal dependencies in upsampled weekly data.

## Installation  
To set up the project locally, follow these steps  
```bash
git clone https://github.com/<your-username>/kitui-crop-yield-ml-prediction.git
cd kitui-crop-yield-ml-prediction
pip install -r requirements.txt
