# Diabetes Risk Assessment Web App

**Project name:** Diabetes Risk Assessment

## Short Introduction

This project is a full-stack web application designed to predict the risk of diabetes in young adults. It features a Python Flask backend that serves a pre-trained machine learning model via a REST API. The frontend is a clean, dependency-free, single-page HTML/CSS/JS application that consumes this API to provide users with a real-time risk assessment based on their health data.

## Features

* **RESTful Backend API:** A robust Flask server that loads a `.pkl` model and provides a `/predict` endpoint.
* **ML Model Integration:** Serves a pre-trained scikit-learn model for real-time predictions.
* **Dependency-Free Frontend:** A single, lightweight HTML file. No `npm`, no React, no build steps required.
* **Interactive UI:** A professional, clinic-themed web form for submitting patient data.
* **Dynamic Results:** Displays the "Diabetic" or "Non-Diabetic" prediction dynamically without a page refresh.
* **Modern Styling:** The UI is fully responsive and styled using Tailwind CSS (loaded via CDN).

## Tech Stack

* **Backend:** Python, Flask, scikit-learn, pandas
* **Frontend:** HTML5, Vanilla JavaScript (ES6+), Tailwind CSS (via CDN)
* **Data:** `.csv` for preprocessing setup, `.pkl` for the trained model

## Setup Instructions

This project consists of a Python backend and a single HTML file for the frontend.

### 1. Clone the Repository

```bash
git clone [https://github.com/madiha522/Diabetes_Risk_Predictor.git](https://github.com/madiha522/Diabetes_Risk_Predictor.git)
cd Diabetes_Risk_Predictor
````

### 2\. Backend Setup (Python Flask)

The backend requires Python and the dependencies listed in `requirements.txt`.

```bash
# Create a virtual environment
python -m venv virtual

# Activate the environment
# On Windows:
.\virtual\Scripts\activate
# On Mac/Linux:
source virtual/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

**IMPORTANT:** The `app.py` server requires the `diabetes_corrected.csv` file to be present in the root directory to set up the data preprocessing steps correctly.

### 3\. Frontend Setup

There is **no setup required** for the frontend. The `Diabetes_frontend.html` file is completely self-contained.

## Running Instructions

You will need to run the backend server and then open the frontend file in your browser.

### 1\. Run the Backend (Terminal)

With your virtual environment activated, run the Flask server:

```bash
python app.py
```

The server will start and be available at `http://127.0.0.1:5000`.

### 2\. Run the Frontend (Browser)

Navigate to the project folder and find the `Diabetes_frontend.html` file.

**Right-click** on the file and select **"Open with"** -\> **"Google Chrome"** (or your preferred browser).

The web page will load, and you can now fill out the form and get predictions from your live backend.

## Project Structure

```
/Diabetes-Risk-Assessment
│
├── app.py                      # The Flask backend server
├── Diabetes_frontend.html         # The single-page HTML/CSS/JS frontend
├── requirements.txt            # Python dependencies
│
├── diabetes_model_final.pkl    # The primary ML model
├── *.pkl                       # Other ML model fallbacks
└── diabetes_corrected.csv      # Required for data column setup
```

## Connect

  * **GitHub:** [github.com/madiha552](https://github.com/)
  * **LinkedIn:** [linkedin.com/in/madiha522](https://www.linkedin.com/)
