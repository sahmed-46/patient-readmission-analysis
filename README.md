# patient-readmission-analysis
The main goal of this project is to analyze and predict hospital readmission based on hospital admission data. Furthermore, a secondary goal is to assess whether an initial diabetes diagnosis can act as a reliable predictor of hospital readmission.

## Outcomes dashboard

Run the interactive dashboard from this directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard filters the included `hospital_readmissions.csv` dataset by age band, medical specialty, diabetes medication, medication changes, A1C and glucose testing, and length of stay. The cohort view shows descriptive outcomes; the model results view shows validation metrics, confusion matrix, coefficients, and predictions from the logistic regression model. Filters are available in both views and select a validation subgroup for model metrics and predictions. These results do not establish causation or predict individual patient outcomes outside the validation set.
