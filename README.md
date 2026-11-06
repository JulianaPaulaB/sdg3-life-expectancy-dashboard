# 🫀 SDG 3: Drivers of Life Expectancy Across Countries

### An Interactive Data Analytics Dashboard Using Regression Analysis

An exploratory data analytics project examining the relationship between life expectancy and selected socioeconomic and environmental indicators in the **United Kingdom and United States from 2000 to 2024**.

Developed as part of **Analytics Techniques and Tools**.

## 🌐 Live Dashboard

Explore the interactive dashboard:

🔗 **[Launch SDG 3: Life Expectancy Dashboard](https://sdg3-life-expectancy-dashboard-78h7lhgxxm3uormh8skbtu.streamlit.app/)**

---

## 📌 Project Overview

Good health and well-being are essential to sustainable development. However, life expectancy is influenced by several interconnected factors, including healthcare spending, economic conditions, environmental quality, and access to basic resources.

This project analyzes selected indicators to identify their relationships with life expectancy and determine which factors have statistically significant associations.

Using regression analysis and an interactive Streamlit dashboard, the project presents data-driven insights that support a better understanding of population health outcomes in relation to **Sustainable Development Goal 3 (SDG 3): Good Health and Well-Being.**

## 🎯 Project Objectives

- Analyze life expectancy trends in the United Kingdom and United States.
- Examine the relationship between life expectancy and selected indicators.
- Identify significant predictors through regression analysis.
- Evaluate model performance using R² and RMSE.
- Present findings through an interactive dashboard.
- Develop data-driven insights related to SDG 3.

## 📊 Key Indicators

| Indicator | Description |
|---|---|
| Life Expectancy | Average expected years of life |
| Healthcare Spending | Healthcare expenditure per capita |
| GDP per Capita | Economic output per person |
| CO₂ Emissions | CO₂ emissions per person |
| Water Access | Access to basic drinking water |

## 🛠️ Tools and Technologies

- **Python** – Data processing and statistical analysis
- **Pandas & NumPy** – Data cleaning and manipulation
- **Statsmodels** – Regression modeling and statistical testing
- **Plotly** – Interactive data visualizations
- **Streamlit** – Interactive dashboard development
- **Jupyter Notebook** – Exploratory analysis and model development

## 📈 Analytical Methodology

The project follows a structured data analytics workflow:

1. **Data Collection** – Gathered indicators from the World Bank and Our World in Data.
2. **Data Preparation** – Combined datasets, handled missing values, and prepared the data for analysis.
3. **Exploratory Data Analysis** – Examined trends, comparisons, and correlations.
4. **Regression Analysis** – Applied Ordinary Least Squares (OLS) and Robust Regression using Huber's M-estimator (RLM).
5. **Model Evaluation** – Assessed model performance using R² and RMSE.
6. **Data Visualization** – Developed an interactive dashboard to communicate findings.

### Regression Model

The analysis uses standardized variables to compare the effects of different indicators on life expectancy.

**Predictor Variables:**
- Healthcare Spending
- GDP per Capita
- CO₂ Emissions
- Water Access

## 📊 Dashboard Features

The interactive dashboard includes:

- **Overview** – Life expectancy trends and country comparisons.
- **Key Drivers** – Relationships between individual indicators and life expectancy.
- **Correlation Analysis** – Pearson correlation coefficients.
- **Regression Model** – Model performance, standardized coefficients, and statistical significance.
- **Insights and Recommendations** – Data-driven interpretations and suggested actions.

Users can filter the dashboard by year and country to explore the results.

## 🔍 Key Findings

The analysis produced the following insights:

1. **CO₂ Emissions** – Showed the strongest negative association with life expectancy in the regression model.
2. **Healthcare Spending** – Demonstrated a positive and statistically significant association under robust regression.
3. **GDP per Capita** – Did not show a statistically significant independent effect after accounting for the other predictors.
4. **Water Access** – Had limited variation across the two countries, making its independent effect difficult to establish.

### Model Evaluation

The project evaluates model performance using:

- R² Score
- Root Mean Squared Error (RMSE)
- Number of Observations

*Note: The analysis uses 44 observations from two countries. Findings indicate statistical associations, not causation, and should not be generalized to all countries.*

## 📁 Project Structure

```text
sdg3-life-expectancy-dashboard/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── SDG3_Life_Expectancy_Regression.ipynb
├── merged_sdg_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/sdg3-life-expectancy-dashboard.git
```

### 2. Navigate to the Project Folder

```bash
cd sdg3-life-expectancy-dashboard
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser.

## 🌍 Relevance to Sustainable Development Goal 3

This project supports **SDG 3: Good Health and Well-Being**, particularly the broader goal of understanding factors associated with population health and longevity.

By examining healthcare, economic, and environmental indicators, the analysis demonstrates how data analytics can contribute to evidence-based discussions and decision-making in public health.

## ⚠️ Limitations

- The analysis covers only two countries: the United Kingdom and United States.
- The dataset contains 44 observations.
- Some indicators have limited variation.
- Missing data may affect the availability of certain country-year comparisons.
- Regression findings indicate associations rather than causal relationships.

## 👩‍💻 Author

**Juliana Paula T. Binas**  
BS Information Systems – Business Analytics  
West Visayas State University

*Academic Project | Analytics Techniques and Tools*

---

**SDG 3: Good Health and Well-Being**

*Turning data into insights for a healthier future.*
