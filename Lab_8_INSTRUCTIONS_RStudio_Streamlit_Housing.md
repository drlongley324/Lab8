---
editor_options: 
  markdown: 
    wrap: 72
---

# Housing Value Prediction with Deep Neural Networks

# ENCE 2530 \@ UTC

## RStudio + Anaconda + TensorFlow + Streamlit

### Educational Use Only

This project is for teaching and learning purposes only. The predictions
from this model are approximate and should **not** be used for real
property appraisal, tax assessment, lending, or investment decisions.

------------------------------------------------------------------------

## 1. Lab Goal

In this lab, you will:

-   use **Python inside RStudio**
-   train a **deep neural network** with **TensorFlow**
-   use a real Hamilton County housing dataset found from
    <https://www.hamiltontn.gov/_downloadsAssessor/AssessorExportCSV.zip>
-   save the trained model
-   build a simple **Streamlit** web app
-   deploy the app through **GitHub** and **Streamlit Community Cloud**

The final product is a small web app that predicts approximate local
housing value from selected property features. This lab will allow
students to practice developing Deep Neural Network models for civil
engineering appplications using real-world data with assistance of AI
tools.

------------------------------------------------------------------------

## 2. Software You Need

Install the following:

### Anaconda

Use Anaconda to manage Python and packages.

### RStudio

Use RStudio as your development environment.

### GitHub account

You will upload your project to GitHub once your trained your model
locally.

------------------------------------------------------------------------

## 3. Create the Python Environment

Open **Anaconda Prompt** and create a Python 3.9 environment (you may
choose to use the virtual environment already crated for this course
such as myenv, just install tensorflow 2.10 and any necessary
labraries/packages: pandas numpy scikit-learn streamlit joblib
matplotlib within this virtual environment):

``` bash
conda create -n housing_dnn python=3.9 -y
conda activate housing_dnn
```

Install required packages:

``` bash
pip install tensorflow==2.10 pandas numpy scikit-learn streamlit joblib matplotlib
```

Check TensorFlow:

``` bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

You should see version `2.10.0` or very close.

------------------------------------------------------------------------

## 4. Configure RStudio to Use the Python Environment (you may skip this step since we have already configured before)

Open **RStudio**.

In the R console, install and load `reticulate` if needed:

``` r
install.packages("reticulate")
library(reticulate)
```

Tell RStudio to use your conda environment:

``` r
use_condaenv("housing_dnn", required = TRUE)
py_config()
```

Make sure the Python path points to the `housing_dnn` environment.

------------------------------------------------------------------------

## 5. Project Folder Structure

Create a project folder such as:

``` text
housing_dnn_streamlit/
│
├── data/
│   └── Housing_Hamilton_Compressed.csv.gz
├── artifacts/
├── train_model.py
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
└── INSTRUCTIONS.md
```

Put the provided `.gz` housing file inside the `data` folder.

------------------------------------------------------------------------

## 6. Create `requirements.txt`

Create a file named `requirements.txt` with the following contents:

``` txt
tensorflow==2.10.0
pandas
numpy
scikit-learn
streamlit
joblib
matplotlib
```

------------------------------------------------------------------------

## 7. Create `runtime.txt`

Create a file named `runtime.txt`:

``` txt
python-3.9
```

This is important when deploying to Streamlit Community Cloud.

------------------------------------------------------------------------

## 8. Understand the Prediction Target

Your model should predict:

``` text
APPRAISED_VALUE
```

Be careful **not** to use columns that directly reveal or nearly
duplicate the target. For example, if a column is already part of the
appraised value calculation, using it may create **target leakage** and
give unrealistic model performance.

A safer approach is to use physical and descriptive property features
such as:

-   `CALC_ACRES`
-   `YEARBUILT`
-   `SIZEAREA`

------------------------------------------------------------------------

## 9. Suggested Feature Strategy

use a **small, reliable set of features** first.

Recommended starting numerical features:

``` python
NUMERIC_FEATURES = [
    "CALC_ACRES",
    "YEARBUILT",
    "SIZEAREA"
]
```

You may later expand the model if the dataset supports more clean
variables.

------------------------------------------------------------------------

## 10. Create `train_model.py`

Your training script should do the following:

1.  load the `.gz` file
2.  clean the target variable
3.  keep a small set of valid features
4.  remove rows with missing values
5.  split data into train and test sets
6.  scale the input features
7.  train a TensorFlow neural network
8.  evaluate the model
9.  save the model and scaler into `artifacts/`

### Essential ideas for the training script

#### Load the data

``` python
import pandas as pd

df = pd.read_csv("data/Housing_Hamilton_Compressed.csv.gz")
print(df.head())
print(df.columns.tolist())
```

#### Keep only needed columns

``` python
features = ["CALC_ACRES", "YEARBUILT", "SIZEAREA"]
target = "APPRAISED_VALUE"

df = df[features + [target]].copy()
```

#### Remove missing values

``` python
df = df.dropna()
```

#### Separate inputs and target

``` python
X = df[features]
y = df[target]
```

#### Train/test split

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### Scale and normalize the inputs data

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Build the neural network

# You may choose 4 layers: one input two hidden and one output, the numbers indicate \# of neurans

``` python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
```

#### Train the model

``` python
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)
```

#### Evaluate the model

``` python
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print("Test MAE:", mae)
```

#### Save artifacts

``` python
import os
import joblib

os.makedirs("artifacts", exist_ok=True)
model.save("artifacts/housing_model.h5")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(features, "artifacts/feature_names.pkl")
```

After training, the `artifacts` folder should contain:

``` text
artifacts/
├── housing_model.h5
├── scaler.pkl
└── feature_names.pkl
```

------------------------------------------------------------------------

## 11. Recommended Improvement: Predict Log Value (We will skip this step for this lab)

Housing values can vary a lot. A neural network often trains better if
you predict the **log of appraised value** instead of the raw dollar
amount.

### Why this helps

-   reduces the effect of very large values
-   improves training stability
-   often gives more reasonable predictions

### Idea

During training:

``` python
import numpy as np

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
```

Train the model using `y_train_log`.

When predicting in the app, convert back using:

``` python
pred_value = np.expm1(pred_log_value)
```

This is strongly recommended for this lab.

------------------------------------------------------------------------

## 12. Create a Simple and Safe Streamlit App

Your Streamlit app should be **student-friendly** and **avoid
unrealistic inputs**.

### Good design choices

-   use only a few input variables
-   show labels clearly
-   set minimum and maximum values
-   warn users that predictions are approximate
-   format output as dollars

### Example app behavior

The app should ask users for:

-   acreage
-   year built
-   building area

Then it should:

1.  load the model
2.  load the scaler
3.  put the user inputs into a DataFrame
4.  scale the inputs
5.  make a prediction
6.  display the estimated value

### Helpful input limits

Use reasonable bounds such as:

-   acres: `0.01` to `20.0`
-   year built: `1900` to `2026`
-   size area: `300` to `10000`

------------------------------------------------------------------------

## 13. Suggested `app.py` Layout

Your app should include:

``` python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
```

### Page title

``` python
st.title("Hamilton County Housing Value Predictor")
st.caption("Educational use only. Predictions are approximate.")
```

### Input widgets

Use `st.number_input()` with bounds:

``` python
acres = st.number_input("Land area (acres)", min_value=0.01, max_value=20.0, value=0.25, step=0.01)
yearbuilt = st.number_input("Year built", min_value=1900, max_value=2026, value=2000, step=1)
sizearea = st.number_input("Building area (sq ft)", min_value=300, max_value=10000, value=1800, step=50)
```

### Prediction button

``` python
if st.button("Predict"):
    input_df = pd.DataFrame({
        "CALC_ACRES": [acres],
        "YEARBUILT": [yearbuilt],
        "SIZEAREA": [sizearea]
    })
```

Then scale and predict.

If you trained on log values, convert back (skip this part for now):

``` python
pred_log = model.predict(input_scaled, verbose=0)[0][0]
pred_value = np.expm1(pred_log)
```

Display a formatted result:

``` python
st.success(f"Estimated appraised value: ${pred_value:,.0f}")
```

------------------------------------------------------------------------

## 14. Prevent Potential Problem

If you get impossible predictions like `$2`, common causes are:

-   missing or bad training rows
-   input values far outside the training range
-   no scaling in the app
-   different feature order in training and prediction
-   trying to predict raw values instead of log values

### Best practices to avoid this

-   drop missing values before training
-   save the scaler and reuse it in the app
-   save the feature names and keep the same order
-   use reasonable input limits
-   predict `log(APPRAISED_VALUE)` and convert back

------------------------------------------------------------------------

## 15. Run the Training Script

Within RStudio, open your train_model.py, select all codes, click run

Skip the followings

From Anaconda Prompt:

``` bash
conda activate housing_dnn
python train_model.py
```

You should see training progress and a message showing the model and
scaler were saved.

------------------------------------------------------------------------

## 16. Run the Streamlit App Locally (You may skip this step)

After training succeeds, run:

``` bash
streamlit run app.py
```

A browser window should open, usually at:

``` text
http://localhost:8501
```

Test several houses with different sizes and ages.

------------------------------------------------------------------------

## 17. Upload the Project to GitHub

Create a GitHub repository and upload:

``` text
app.py
train_model.py
requirements.txt
runtime.txt
README.md
artifacts/housing_model.h5
artifacts/scaler.pkl
artifacts/feature_names.pkl
```

Depending on file size, you may choose to retrain during deployment or
upload artifacts directly. artifacts is a folder that saves
housing_model.h5, scaler.pkl,and feature_names.pkl

------------------------------------------------------------------------

## 18. Deploy with Streamlit Community Cloud

1.  go to Streamlit Community Cloud
2.  sign in with GitHub
3.  choose your repository
4.  set the main file as `app.py`
5.  deploy

The platform will use:

-   `requirements.txt` to install packages
-   `runtime.txt` to choose Python 3.9

------------------------------------------------------------------------

## 19. What to Submit

Submit the following:

### 1. GitHub repository link

### 2. Streamlit app link

### 3. Short reflection report

Your reflection should explain:

-   what features you used
-   why scaling was needed
-   your model architecture
-   whether your predictions looked reasonable
-   limitations of the model
-   ethical concerns of AI-based valuation

------------------------------------------------------------------------

## 20. Reflection Questions

Answer these in your report:

1.  Why is data cleaning important before training a neural network?
2.  Why is feature scaling important?
3.  Why can real-world housing data be difficult to model?
4.  Why should we avoid target leakage?
5.  Why is this model suitable for education but not for professional
    appraisal?

------------------------------------------------------------------------

## 21. Recommended Teaching Version of the App

For this lab, a **simple three-input Streamlit app** is strongly
recommended:

-   `CALC_ACRES`
-   `YEARBUILT`
-   `SIZEAREA`

Why this is a good choice:

-   easy for students to understand
-   easy to debug
-   low risk of missing-value problems
-   cleaner interface
-   fewer deployment problems

Once this works, you can later extend the project to include more
variables.

------------------------------------------------------------------------

## 22. Final Reminder

This project demonstrates a complete AI workflow:

-   data loading
-   data cleaning
-   neural network training
-   saving model artifacts
-   creating a web app
-   cloud deployment

That workflow is one of the most valuable practical AI skills civil
engineering students can learn. One of the strengths of deep learning
and DNNs is that a solid understanding of the essential concepts is
enough to begin applying the tools to solve challenging civil
engineering problems.

## 23. Grading Rubric

| Category | Description | Points |
|----|----|----|
| **1. Data Processing & Model Training** | Correctly loads the housing dataset, performs basic data cleaning, and trains a TensorFlow neural network model. | **30** |
| **2. Model Results** | Model runs successfully and produces reasonable predictions (training completed, evaluation metric reported). | **25** |
| **3. Streamlit Prediction App** | Creates a working Streamlit app that allows users to input housing features and obtain predicted values. | **25** |
| **4. Deployment & Documentation** | Project uploaded to GitHub and deployed through Streamlit Cloud. Includes a brief reflection or README describing the workflow and limitations. | **20** |
