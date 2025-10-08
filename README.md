# 📈 Stock Prediction Dashboard

A user-friendly and interactive web dashboard for visualizing historical stock data and forecasting future prices using the Prophet time-series model. Built with Streamlit, Prophet, and yfinance.

## 📊 Project Overview

This project is an interactive web dashboard designed for visualizing historical stock prices and forecasting future trends. Built with Python, Streamlit, and Facebook's Prophet model, it provides a user-friendly interface to fetch, analyze, and predict data for any stock ticker available on Yahoo Finance.

The dashboard allows users to dynamically select a stock, define a historical date range, and specify the number of future days to forecast. The results are presented in clean, interactive charts, separating historical data from the predictive forecast, complete with confidence intervals.

### Key Features

* **Dynamic Data Fetching**: Pulls real-time and historical stock data directly from Yahoo Finance.
* **Predictive Forecasting**: Utilizes the robust Prophet time-series model to generate future price predictions.
* **Interactive Controls**: Users can easily customize the stock symbol, date range, forecast duration, and seasonality parameters through a simple sidebar menu.
* **Rich Visualizations**: Employs Plotly to create beautiful, interactive charts with zoom, pan, and hover capabilities.
* **Data Export**: Allows users to download the generated forecast data as a CSV file for further offline analysis.
* **Clean UI**: A dual-tab interface neatly separates the historical data view from the forecast results for a clear and intuitive user experience.

### Technology Stack

* **Core Language**: Python
* **Web Framework**: Streamlit
* **Forecasting Model**: Prophet
* **Data Source**: yfinance
* **Visualization**: Plotly
* **Data Manipulation**: Pandas & NumPy

## 🌟 Key Features

* **Interactive UI**: A clean and simple interface built with Streamlit.
* **Dynamic Data Fetching**: Pulls historical stock data directly from Yahoo Finance using `yfinance`.
* **Powerful Forecasting**: Utilizes Facebook's `Prophet` library to generate robust time-series forecasts.
* **Customizable Inputs**:
    * Select from popular stocks (`GOOGL`, `AAPL`, `MSFT`, etc.) or enter any custom stock ticker.
    * Define a custom date range for historical data analysis.
    * Adjust the forecast period from 30 to 1095 days.
    * Choose the data interval (`daily`, `weekly`, `monthly`).
* **Rich Visualizations**: Interactive charts powered by Plotly, showing historical trends, forecasts, and confidence intervals.
* **Data Export**: Download the generated forecast data as a CSV file for further analysis.
* **Custom Theming**: Toggle between light and dark modes for better viewing comfort.

## 🛠️ Technologies Used

* **Python**: The core programming language.
* **Streamlit**: For building and deploying the interactive web application.
* **Prophet**: For time-series forecasting.
* **yfinance**: For fetching financial data from Yahoo Finance.
* **Plotly**: For creating interactive and beautiful data visualizations.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.

## 📦 Installation & Setup

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

* Python 3.8 or newer
* `pip` (Python package installer)

### 2. Clone the Repository

Clone this repository to your local machine:
```bash
git clone https://github.com/Shubh-Dodiya/Stock-Prediction-Web-application.git
```

## 📁 File Structure

Here's an overview of the project's directory structure:
```text
stock-prediction-dashboard/
├── README.md           # The project explanation you've created
├── requirements.txt    # Lists all Python package dependencies
└── stock-prediction.py # Your main Streamlit application script
```
