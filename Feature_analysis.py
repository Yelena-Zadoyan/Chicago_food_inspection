import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feature_analysis(data):
    # VIOLATION FEATURE ANALYSIS
    data['Violation_NA_Count'] = data.groupby('License #')['Violations'].apply(lambda x: x.isna().sum())
    data['Violation_NA_Count'].fillna(0, inplace=True)
    data['Violation_NA_Percentage'] = (data['Violation_NA_Count'] / data.groupby('License #').size()) * 100
    # dropping the rows with violations missing values
    data.drop(data[data['Violation_NA_Percentage'] > 0].index, axis=0, inplace=True)
    data.drop(['Violation_NA_Count', 'Violation_NA_Percentage'], axis=1, inplace=True)
    # Calculate the count of violations per License #
    data['violations_count_per_license'] = data.groupby('License #')['Violations'].transform('count')
    # Calculate the average count of violations per establishment
    data['average_violations_per_establishment'] = data['violations_count_per_license'] / data.groupby('License #')[
        'License #'].transform('count') * 100
    data['fail_rate_per_establishment'] = data.groupby('License #')['Fail'].transform('sum') / \
                                          data.groupby('License #')['License #'].transform('count') * 100

    # Scatter plot_ by violation count
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='violations_count_per_license', y='fail_rate_per_establishment', data=data)

    plt.title('Scatter Plot of violations_count_per_license & fail rate')
    plt.xlabel('violations_count_per_license')
    plt.ylabel('fail_rate_per_establishment')
    plt.show()

    # Scatter plot - by violation weight
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='average_violations_per_establishment', y='fail_rate_per_establishment', data=data)

    plt.title('Scatter Plot of Average Violations per Establishment vs. fail_rate_per_establishment')
    plt.xlabel('Average Violations per Establishment')
    plt.ylabel('fail_rate_per_establishment')
    plt.show()


    # Dropping Outlier with high violation
    data.drop(data[data['violations_count_per_license'] > 50].index, inplace=True)
    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='violations_count_per_license', y='fail_rate_per_establishment', data=data)

    plt.title('Scatter Plot of violations_count_per_license & fail rate')
    plt.xlabel('violations_count_per_license')
    plt.ylabel('fail_rate_per_establishment')
    plt.show()

    # INSPECTION TYPE FEATURE ANALYSIS

    # dropping the 1 na row with missing inspection type
    # data['Inspection Type'].isna().sum()
    data = data.drop(data[data['Inspection Type'].isna()].index)
    # selecting the data that contain the specified inspection types
    data['Inspection_Canvass'] = data['Inspection Type'].str.contains('Canvass', case=False, na=False)
    data['Inspection_Suspect'] = data['Inspection Type'].str.contains('Suspect', case=False, na=False)
    data['Inspection_Task'] = data['Inspection Type'].str.contains('Task', case=False, na=False)
    data['Inspection_Consultation'] = data['Inspection Type'].str.contains('Consultation', case=False, na=False)
    data['Inspection_Complaint'] = data['Inspection Type'].str.contains('Complaint', case=False, na=False)
    data['Inspection_License'] = data['Inspection Type'].str.contains('License', case=False, na=False)
    data['Inspection_other'] = ~data['Inspection Type'].str.contains(
        'Canvass|Suspect|Task|Consultation|Complaint|License', case=False, na=False)

    # fail rate analysis by inspection types
    print('Fail rate analysis by inspection types')
    fail_rate_per_canvas = data.groupby('Inspection_Canvass')['Fail'].sum() / data.groupby('Inspection_Canvass')[
        'Inspection_Canvass'].sum() * 100
    fail_rate_per_suspect = data.groupby('Inspection_Suspect')['Fail'].sum() / data.groupby('Inspection_Suspect')[
        'Inspection_Suspect'].sum() * 100
    fail_rate_per_Task = data.groupby('Inspection_Task')['Fail'].sum() / data.groupby('Inspection_Task')[
        'Inspection_Task'].sum() * 100
    fail_rate_per_Consultation = data.groupby('Inspection_Consultation')['Fail'].sum() / \
                                 data.groupby('Inspection_Consultation')['Inspection_Consultation'].sum() * 100
    fail_rate_per_complaint = data.groupby('Inspection_Complaint')['Fail'].sum() / data.groupby('Inspection_Complaint')[
        'Inspection_Complaint'].sum() * 100
    fail_rate_per_license = data.groupby('Inspection_License')['Fail'].sum() / data.groupby('Inspection_License')[
        'Inspection_License'].sum() * 100
    fail_rate_per_other = data.groupby('Inspection_other')['Fail'].sum() / data.groupby('Inspection_other')[
        'Inspection_other'].sum() * 100

    print(fail_rate_per_canvas, fail_rate_per_suspect, fail_rate_per_Task, fail_rate_per_Consultation,
          fail_rate_per_complaint, fail_rate_per_license, fail_rate_per_other)

    # RISK FEATURE ANALYSIS
    fail_percentage_by_risk = (data.groupby('Risk')['Fail'].sum() / data.groupby('Risk').size()) * 100

    Failure_by_risk = pd.DataFrame({'Risk': fail_percentage_by_risk.index, 'Failure': fail_percentage_by_risk})

    plt.figure(figsize=(20, 6))
    sns.barplot(x='Risk', y='Failure', data=Failure_by_risk)

    plt.title('Fail Percentage by Risk')
    plt.xlabel('Risk')
    plt.ylabel('Fail Percentage')
    plt.xticks(rotation=90)
    plt.show()

    # TYPEOF FACILITY FEATURE
    Facility_counts = data['Facility Type'].value_counts().sort_values(ascending=False)
    fail_percentage_by_facility = data.groupby('Facility Type')['Fail'].sum() / data.groupby(
        'Facility Type').size() * 100
    Failure_by_facility = pd.DataFrame(
        {'Facility Type': fail_percentage_by_facility.index, 'Failure': fail_percentage_by_facility.values,
         'Num_Inspections': Facility_counts.values})
    Failure_by_facility = Failure_by_facility.sort_values(by='Facility Type')

    # Set up the figure and the first y-axis (fail percentage)
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1 = sns.barplot(x='Facility Type', y='Failure', data=Failure_by_facility, ax=ax1, color='steelblue')
    ax1.set_ylabel('Fail Percentage', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Create the second y-axis (number of inspections) and plot the bar graph
    ax2 = ax1.twinx()
    ax2 = sns.barplot(x='Facility Type', y='Num_Inspections', data=Failure_by_facility, ax=ax2, color='lightcoral')
    ax2.set_ylabel('Number of Inspections', color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='lightcoral')

    plt.title('Fail Percentage and Number of Inspections by Facility')
    plt.xlabel('Facility')
    plt.xticks(rotation=90)
    plt.show()

    facility_fail_count = data.groupby('Facility Type')['Fail'].sum().reset_index()
    facility_fail_count = facility_fail_count[facility_fail_count['Fail'] > 10].sort_values(by='Fail', ascending=False)
    print(facility_fail_count)

    print('Facility weight in total establishments')
    facility_weight = Facility_counts / len(data['Facility Type']) * 100
    facility_weight.sort_values(ascending=False)
    print(facility_weight)

    # DUMMIES

    # Risk dummies
    data = pd.concat([data, pd.get_dummies(data['Risk'], prefix='Risk', prefix_sep='_')], axis=1)
    # Filter facility types where the weight is greater than 5%
    selected_facility_types = facility_weight[facility_weight > 5].index
    # Create dummy variables only for the selected facility types
    dummy_columns = [f'Facility Type_{facility_type}' for facility_type in selected_facility_types]
    data = pd.concat(
        [data, pd.get_dummies(data['Facility Type'], prefix='Facility Type', prefix_sep='_')[dummy_columns]], axis=1)
    # Filter facility types where the weight is lower than 5%
    other_facility_types = facility_weight[facility_weight <= 5].index
    # Create a new column 'Facility Type_other' with 1 if it's another facility type, 0 otherwise
    data['Facility Type_other'] = data['Facility Type'].apply(lambda x: 1 if x in other_facility_types else 0)
    # Drop the original 'Facility Type' column
    data.drop(columns='Facility Type', inplace=True)

    # Drop rows with missing license numbers
    data.dropna(subset=['License #'], inplace=True)


    # columns_to_drop = ['DBA Name', 'AKA Name', 'Address', 'City', 'State', 'Latitude', 'Longitude', 'Location']
    columns_to_drop = ['DBA Name', 'AKA Name', 'Address', 'City', 'State', 'Latitude', 'Longitude', 'Location',
                       'Inspection Type', 'Risk', 'Zip', 'Violations',  'Inspection ID', 'License #',
                       'Inspection Date', 'violations_count_per_license', 'fail_rate_per_establishment']
    data.drop(columns_to_drop, axis=1, inplace=True)

    return data





