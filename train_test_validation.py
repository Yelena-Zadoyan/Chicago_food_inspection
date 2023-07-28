def model(data):
    # Split the Data
    from sklearn.model_selection import train_test_split

    # Split the data into features (X) and target (y)
    X = data.drop(columns=['Fail'])
    y = data['Fail']

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Step 3: Choose Candidate Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    # Initialize candidate models
    logistic_regression = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000, random_state=42)
    decision_tree = DecisionTreeClassifier(random_state=42)
    random_forest = RandomForestClassifier(random_state=42)
    gradient_boosting = GradientBoostingClassifier(random_state=42)

    # Step 4: Train and Tune Models
    from sklearn.model_selection import GridSearchCV

    # Define hyperparameter grids for tuning
    param_grid_logreg = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
    }

    param_grid_dt = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
    }

    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.01, 0.001],
    }

    # Perform GridSearchCV for each model to find the best hyperparameters
    grid_search_logreg = GridSearchCV(logistic_regression, param_grid_logreg, cv=5)
    grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5)
    grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5)
    grid_search_gb = GridSearchCV(gradient_boosting, param_grid_gb, cv=5)

    # Fit the models with the training data
    grid_search_logreg.fit(X_train, y_train)
    grid_search_dt.fit(X_train, y_train)
    grid_search_rf.fit(X_train, y_train)
    grid_search_gb.fit(X_train, y_train)

    # Step 5: Evaluate Models
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Define a function to evaluate the models
    def evaluate_model(model, X, y):
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        return accuracy, precision, recall, f1

    # Evaluate the models on the training set
    logreg_accuracy_train, logreg_precision_train, logreg_recall_train, logreg_f1_train = evaluate_model(
        grid_search_logreg, X_train, y_train)
    dt_accuracy_train, dt_precision_train, dt_recall_train, dt_f1_train = evaluate_model(
        grid_search_dt, X_train, y_train)
    rf_accuracy_train, rf_precision_train, rf_recall_train, rf_f1_train = evaluate_model(
        grid_search_rf, X_train, y_train)
    gb_accuracy_train, gb_precision_train, gb_recall_train, gb_f1_train = evaluate_model(
        grid_search_gb, X_train, y_train)

    # Evaluate the models on the validation set
    logreg_accuracy_val, logreg_precision_val, logreg_recall_val, logreg_f1_val = evaluate_model(
        grid_search_logreg, X_val, y_val)
    dt_accuracy_val, dt_precision_val, dt_recall_val, dt_f1_val = evaluate_model(grid_search_dt, X_val, y_val)
    rf_accuracy_val, rf_precision_val, rf_recall_val, rf_f1_val = evaluate_model(grid_search_rf, X_val, y_val)
    gb_accuracy_val, gb_precision_val, gb_recall_val, gb_f1_val = evaluate_model(grid_search_gb, X_val, y_val)

    # Evaluate the models on the testing set
    logreg_accuracy_test, logreg_precision_test, logreg_recall_test, logreg_f1_test = evaluate_model(
        grid_search_logreg, X_test, y_test)
    dt_accuracy_test, dt_precision_test, dt_recall_test, dt_f1_test = evaluate_model(grid_search_dt, X_test, y_test)
    rf_accuracy_test, rf_precision_test, rf_recall_test, rf_f1_test = evaluate_model(grid_search_rf, X_test, y_test)
    gb_accuracy_test, gb_precision_test, gb_recall_test, gb_f1_test = evaluate_model(grid_search_gb, X_test, y_test)

    # Print the evaluation metrics for train model
    print("Logistic Regression - Training sample:")
    print(
        f"Accuracy: {logreg_accuracy_train}, Precision: {logreg_precision_train}, Recall: {logreg_recall_train}, F1: {logreg_f1_train}")

    print("Decision Tree - Training sample:")
    print(
        f"Accuracy: {dt_accuracy_train}, Precision: {dt_precision_train}, Recall: {dt_recall_train}, F1: {dt_f1_train}")

    print("Random Forest - Training sample:")
    print(
        f"Accuracy - Training sample: {rf_accuracy_train}, Precision: {rf_precision_train}, Recall: {rf_recall_train}, F1: {rf_f1_train}")

    print("Gradient Boosting - Training sample:")
    print(
        f"Accuracy: {gb_accuracy_train}, Precision: {gb_precision_train}, Recall: {gb_recall_train}, F1: {gb_f1_train}")

    # Print the evaluation metrics for validation model
    print("Logistic Regression - Validation sample:")
    print(
        f"Accuracy: {logreg_accuracy_val}, Precision: {logreg_precision_val}, Recall: {logreg_recall_val}, F1: {logreg_f1_val}")

    print("Decision Tree - Validation sample:")
    print(
        f"Accuracy: {dt_accuracy_val}, Precision: {dt_precision_val}, Recall: {dt_recall_val}, F1: {dt_f1_val}")

    print("Random Forest - Validation sample:")
    print(
        f"Accuracy - Val sample: {rf_accuracy_val}, Precision: {rf_precision_val}, Recall: {rf_recall_val}, F1: {rf_f1_val}")

    print("Gradient Boosting - Validation sample:")
    print(
        f"Accuracy: {gb_accuracy_val}, Precision: {gb_precision_val}, Recall: {gb_recall_val}, F1: {gb_f1_val}")

    # Print the evaluation metrics for testing sample
    print("Logistic Regression - Testing sample:")
    print(
        f"Accuracy: {logreg_accuracy_test}, Precision: {logreg_precision_test}, Recall: {logreg_recall_test}, F1: {logreg_f1_test}")

    print("Decision Tree - Testing sample:")
    print(
        f"Accuracy: {dt_accuracy_test}, Precision: {dt_precision_test}, Recall: {dt_recall_test}, F1: {dt_f1_test}")

    print("Random Forest - Testing sample:")
    print(
        f"Accuracy - Testing sample: {rf_accuracy_test}, Precision: {rf_precision_test}, Recall: {rf_recall_test}, F1: {rf_f1_test}")

    print("Gradient Boosting - Testing sample:")
    print(
        f"Accuracy: {gb_accuracy_test}, Precision: {gb_precision_test}, Recall: {gb_recall_test}, F1: {gb_f1_test}")


    #ROC-AUC curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    # Define a function to plot the ROC curve and calculate AUC
    def plot_roc_auc(model, X_train, y_train, model_name):
        y_prob = model.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, y_prob)
        auc_score = roc_auc_score(y_train, y_prob)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random model)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)

    # Call the function for each model
    plot_roc_auc(grid_search_logreg, X_train, y_train, 'Logistic Regression')
    plot_roc_auc(grid_search_dt, X_train, y_train, 'Decision Tree')
    plot_roc_auc(grid_search_rf, X_train, y_train, 'Random Forest')
    plot_roc_auc(grid_search_gb, X_train, y_train, 'Gradient Boosting')

    plt.show()

    # Print the best parameters for each model
    print("Logistic Regression - Best Parameters:")
    print(grid_search_logreg.best_params_)

    print("Decision Tree - Best Parameters:")
    print(grid_search_dt.best_params_)

    print("Random Forest - Best Parameters:")
    print(grid_search_rf.best_params_)

    print("Gradient Boosting - Best Parameters:")
    print(grid_search_gb.best_params_)
