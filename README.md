# PROJECT TITLE
## **Ames House Price Prediction**

## Project Description
In this project we have build and deployed a machine learning model to predict house prices using the Ames Housing dataset.

## Dataset Used:

This project uses the **Ames Housing Dataset**, which contains detailed
information about residential homes in Ames, Iowa.

The dataset includes features(82) some of them are:
(In the UI it was not possible(headache) to enter every feature for an user so i compressed it to least feature, where the other features will get the average value accordingly):

- Lot Area
- Overall Quality
- Year Built
- Living Area
- Garage Capacity
- Neighborhood

## Model Performance:
Evaluation metrics obtained during training:<br>
for XGBoost:<br>
-R² Score: 0.90<br>
-MAE: 12231<br>

for SVR:<br>
-R² Score: 0.86<br>
-MAE: 13605<br>

for RidgeCV:<br>
-R² Score: 0.84<br>
-MAE: 15399<br>
<br>
(We choosed the best model for deployment..i.e XGBoost)
<br>
_The system includes:_<br>
• Data preprocessing<br>
• Feature engineering<br>
• XGBoost regression model<br>
• Flask API for inference<br>
• Web interface for user input<br>
• Cloud deployment using Render<br>
<br>

## Architecture (Workflow):<br>
User Input  
↓  
Flask API  
↓  
Data Preprocessing  
↓  
XGBoost Model  
↓  
Prediction  
↓  
Displayed in Web UI



## Tech Stack Used<br>
1.Python<br>
2.Flask<br>
3.XGBoost<br>
4.Scikit_learn<br>
5.Pandas<br>
6.HTML/CSS<br>
7.Render(Deployment)<br>

## Live Demo(Model Link)<br>

**https://ameshousepricing.onrender.com**<br>

## Screenshot Section<br>
![web Interface](https://github.com/Surya117117/ameshousepricing/blob/8d3ed2b7b4a1a775c1ef1e3a8fbbf187a0e6cf77/Screenshot%202026-03-16%20183950.png)
ameshousepricing

## Project Structure

ameshousepricing/<br>
│<br>
├── model/<br>
│ ├── xgb_model.pkl<br>
│ └── model_columns.pkl<br>
│<br>
├── templates/<br>
│ └── index.html<br>
│<br>
├── static/<br>
│ └── style.css<br>
│<br>
├── app.py<br>
├── requirements.txt<br>
├── README.md<br>
└── notebook.ipynb<br>

