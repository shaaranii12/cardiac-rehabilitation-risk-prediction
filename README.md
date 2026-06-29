# Cardiac Rehabilitation Risk Prediction 

### Live Demo  
🔗 Try it here: [CR Risk Predictor](https://huggingface.co/spaces/Shaaranii12/emotion-analyzer)  

## Overview  
Cardiac rehabilitation is a structured program designed to improve the health and recovery of patients with cardiovascular disease. Identifying patients who may be at elevated risk during rehabilitation allows clinicians to provide more personalized care and exercise recommendations.

This project was developed in collaboration with University Malaya Medical Centre (UMMC) as my Final Year Project.

The system predicts patient risk using clinical, demographic, lifestyle, and exercise performance data while providing interpretable explanations for each prediction through Explainable AI (SHAP).

## Features  
✅ Predicts cardiac rehabilitation risk using a CatBoost machine learning model.<br>
✅ Provides patient-specific risk probabilities.<br>
✅ Explains model predictions using SHAP values.<br>
✅ Highlights protective and risk-increasing factors influencing each prediction.<br>
✅ Interactive Streamlit web application for clinicians and researchers.<br>
✅ User-friendly interface for exploring individual patient predictions.<br>


### AI Clinical Assistant 🤖 
Beyond risk prediction, the system incorporates a Large Language Model (LLM) to transform model outputs into clear, patient-specific clinical insights.

The AI assistant:<br>
✅ Generates personalized risk assessment reports. <br>
✅ Explains the clinical significance of key risk factors. <br>
✅ Summarizes protective and residual risk factors in natural language. <br>
✅ Enhances interpretability without altering the underlying prediction model.

## Model Development & Evaluation
* Evaluated and compared **CatBoost, XGBoost, Random Forest (RF), and Artificial Neural Network (ANN)** for cardiac rehabilitation risk prediction.
* Performed iterative **feature selection** to reduce model complexity while improving predictive performance.
* Compared models before and after feature selection using **ROC-AUC** and other classification metrics.
* Selected **CatBoost** as the final model based on its overall performance and robustness.

**Feature selection reduced model complexity while maintaining strong predictive performance and improving model generalization.** <br>
![Impact of Feature Selection](images/Graph1.png)  



## Screenshots
The model was fine-tuned using the j-hartmann/emotion-english-distilroberta-base transformer, and evaluated to measure its performance across different emotion categories

### Confusion Matrix
![Confusion Matrix](images/ConfusionMatrix.png)  

### Input and Prediction Examples
![Demo 1](images/demo1.png)
![Demo 2](images/demo2.png)  

## Tools and Technologies used 
- **Python**  
- **SHAP (Explainable AI)**
- **Scikit-learn**
- **Pandas & NumPy** 
- **Streamlit** 

## Author
**Shaarani Navaratnam**
<br> Data Science Student 
<br> University Malaya
