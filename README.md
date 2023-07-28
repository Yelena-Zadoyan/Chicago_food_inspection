# Chicago_food_inspection
Evaluates the failure rate during the inspection based  on different features.

# Data
The database contains from the establishment name, address, location coordinates, type, inspection description, dates,
the risk level, inspection results and violation types.

Separate feature analysis is conducted, based on which the ony features that are left for model consideration are the
following: target variable the failure as an outcome of the inspection, and the features - violation rate per
establishment, inspection types, risk level, type of activity

# Model
Logistic regression, Decision trees, Random Forest, Gradient Boosting

# Structure
Main.py file, README.md and 2 files with callable functions: one for feature analysis and the other for train, test
and validation sample selection, model building and fitting.

# Results
All the models have high accuracy and precision rates and very low recall and F1 score. Thus, the results can be used
mostly in case of Resource Allocation issues, when limited resources are directed toward inspections that are more
likely to identify actual failures, reducing unnecessary inspections on compliant establishments, and for Legal and
Regulatory compliance issues, when precision might be prioritized to minimize the risk of wrongly penalizing compliant
establishments. But the model cannot be considered in case of Health and Safety concerns.
