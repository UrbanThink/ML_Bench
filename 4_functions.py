# CSP ML Algorithm
# CSP = Categorical - Similarity - Probability
# This function makes the full model (with all data) and the train model (with partial data : split train-test) and evaluates the train model.
def csp():
    # Create the train, test and full sets.
    # The train set (X_train) is df without the x% test rows extracted. X_train contains the values to be predicted (last column).
    # The test set (X_test) is a random extract of x% of df rows. X_test does not contain the values to be predicted (last column removed).
    # The full train set (X_full_train) is full df, used to make the model with all data. X_full_train contains the values to be predicted (last column).
    # The full test set (X_full_test) is full df, used to make the model with all data. X_full_test does not contain the values to be predicted (last column removed).
    X_full_train = deepcopy(df)
    X_full_test = deepcopy(df)
    X_train = deepcopy(df)
    X_test = []
    # Feed X_test by extracting from X_train
    rows_qty_to_extract = round(len(df) * 0.005)
    for i in range(0, rows_qty_to_extract):
        random_index = random.randint(0, len(X_train)-1)
        X_test.append(X_train[random_index])
        del X_train[random_index]
    # Split X_test into X_test and y_test
    y_test = []
    for tab in X_test:
        y_test.append(tab[-1])
        del tab[-1]
    # Split X_full_test into X_full_test and y_full_test
    y_full_test = []
    for tab in X_full_test:
        y_full_test.append(tab[-1])
        del tab[-1]  
    
    # --------------------
    # Make Model for splitted data
    print("Making Model (for splitted data)...")
    csp_model_maker(X_train, "train")
    
    # Evaluate model for splitted data
    # Predict on test data
    y_pred = ask_csp(X_test, "train")
    # Compute evaluation metrics
    evaluation_metrics = evaluate_model(y_test, y_pred)
    # --------------------
    # # Make Model for full data
    # print(LINE_UP, end=LINE_CLEAR)
    # print("Making Model (for full data)...")
    # csp_model_maker(X_full_train, "full")
    
    # # Evaluate model for full data
    # # Predict on test data
    # y_full_pred = ask_csp(X_full_test, "full")
    # # Compute evaluation metrics
    # evaluation_metrics = evaluate_model(y_full_test, y_full_pred)
    # --------------------
    
    return [
        evaluation_metrics
    ]



# Function to make the csp models.
# Input X_train : data to work with, either full or just a train part.
# Input mode : "full" (all data) or "train" (partial data).
def csp_model_maker(X_train, mode):   
    
    # ------------------------------
    X_train_rows = len(X_train)
    X_train_columns = len(X_train[0])
    X_train_max_index = X_train_columns - 1   


    # ------------------------------
    # Create model_rows_root
    # ------------------------------
    
    # Create a model associating to each possible row, its possible predicted values with their quantities observed in X_train.
    # Ex: string_row_x   => pv1 => 2
    #                    => pv5 => 3
    model_rows_root = {}
    for r in range(0, X_train_rows):
        list_row = deepcopy(X_train[r])
        list_row.pop()
        
        # Stringify the row
        # "@@@" is the separator between values
        row = ""
        for val in list_row:
            row += str(val) + "@@@"
        
        # If the row already exists in the model.
        if row in model_rows_root:
            # Get the pv of the row.
            pv_to_append = X_train[r][X_train_max_index]
            # If the pv is in mrr for the row.
            if pv_to_append in model_rows_root[row]:
                # Increment the pv value for this row in mrr.
                model_rows_root[row][pv_to_append] += 1
            # Else, create a new key-value to add the pv for this row in mrr.
            else:
                new_key_value = {pv_to_append:1}
                model_rows_root[row].update(new_key_value)
        # Else, create a new key-value to add the row to mrr with its pv.
        else:
            new_key_value = {row:{}}
            model_rows_root.update(new_key_value)
            pv_to_append = X_train[r][X_train_max_index]
            new_key_value = {pv_to_append:1}
            model_rows_root[row].update(new_key_value)
    
    
    # ------------------------------
    # Create model_complete and model_reducted  
    # ------------------------------
    
    # Explode X_train in many parts, each part corresponding to a possible value to predict (Y) and containing the rows associated to this value.
    df_ex = {}
    predict_values = []
    for r in range(0, X_train_rows):
        predict_value = X_train[r][X_train_max_index]
        if predict_value in predict_values:
            row_to_append = deepcopy(X_train[r])
            row_to_append.pop()
            df_ex[predict_value].append(row_to_append)
        else:
            predict_values.append(predict_value)
            new_key_value = {predict_value:[]}
            df_ex.update(new_key_value)
            row_to_append = deepcopy(X_train[r])
            row_to_append.pop()
            df_ex[predict_value].append(row_to_append)
    
    # ------------------------------
    # Retrieve the values appearing for each predictor field.
    # Ex : {1, 3}{2, 3} => predicted_value_1
    # Each {} contains the possible values of a predictor field that are present in the data for the value to be predicted currently being processed.
    # Associate the percentage of presence with each value of the super key (the total number of occurrences is the number of occurrences for the value to be predicted currently being processed).
    # Ex : {1:0.3, 3:0.1}{2:0.45, 3:0.1} => predicted_value_1
    # So, a complete model with 3 pvs would look like this :
    # {1:0.3, 3:0.1}{2:0.45, 3:0.1} => predicted_value_1
    # {3:0.1}{1:0.2, 2:0.5} => predicted_value_2
    # {2:0.1}{3:0.7} => predicted_value_3
    # ------------------------------
    # Number of possible predict values.
    n_pv = len(df_ex)
    model_complete = {}
    for key in df_ex:
        # Number of rows for this pv.
        n_pv_rows = len(df_ex[key])
        
        # --------------------
        # Calculation of the percent_value, which is the base value that will be added to the associated model values.
        
        # 1st Version => Abandoned
        # --------------------
        # percent_value = 1/n_pv_rows
        # --------------------
        
        # 2nd Version => TOP
        # --------------------
        # coef = (n_pv_rows/X_train_rows)
        # percent_value = ((1/n_pv_rows) + ((1+coef)/X_train_rows))
        # --------------------
        # 2nd Version Simplified
        # --------------------
        percent_value = (1/n_pv_rows) + (1/X_train_rows) + (n_pv_rows/X_train_rows**2)
        # --------------------
        # This formula means that at the end of the model, each value is associated with: its probability for the pv to which it belongs plus its number of occurrences for the pv relative to the population size of the global data plus its number of occurrences for the pv multiplied by the population size of the pv and relative to the squared population size of the global data.
        # More simply, in fine in the model, each value is associated with: its probability for the pv plus its number of occurrences coefficient for the pv relative to the population size of the global data.
        # --------------------
        
        # 3rd Version => TOP
        # --------------------
        # coef = (n_pv_rows/X_train_rows)
        # percent_value = (((1+coef)/n_pv_rows) + (1/X_train_rows))
        # --------------------
        # 3rd Version Simplified
        # --------------------
        # percent_value = (1/n_pv_rows) + (2/X_train_rows)
        # --------------------
        # This formula means that in the model, each value is associated with: its probability for the pv to which it belongs, plus twice its number of occurrences for the pv and relative to the population size of the global data.
        # In other words, each value in the model is associated with: its coefficient probability for the pv to which it belongs plus its number of occurrences for the pv and relative to the population size of the global data.
        # --------------------

        # 4th Version => Abandoned
        # --------------------
        # percent_value = (1/n_pv_rows) * (2/X_train_rows)
        # --------------------
        
        # 5th Version => Abandoned
        # --------------------
        # percent_value = (1/n_pv_rows) * (1/X_train_rows) * (n_pv_rows/X_train_rows**2)
        # --------------------
        
        # Add pair key_value to model_complete.
        new_key_value = {key:{}}
        model_complete.update(new_key_value)
        # Go through this pv rows.
        for row in range(0, n_pv_rows):
            for column in range(0, X_train_columns-1):
                if row == 0:
                    new_key_value = {column:{}}
                    model_complete[key].update(new_key_value)
                column_value = df_ex[key][row][column]
                if column_value in model_complete[key][column]:
                    model_complete[key][column][column_value] += percent_value
                else:
                    new_key_value = {column_value:percent_value}
                    model_complete[key][column].update(new_key_value)
    
    # ------------------------------
    # Calculate the reduced model by removing from the full model all values with a presence percentage < reduc_value
    reduc_value = 0.1
    model_reducted = deepcopy(model_complete)
    for key in model_complete:
        for column in model_complete[key]:
            for value in model_complete[key][column]:
                if model_complete[key][column][value] < reduc_value:
                    del model_reducted[key][column][value]

    # ------------------------------
    # Save models for future queries using ask_csp()
    if mode == "train":
        with open('Outputs\csp_model_complete_train.pkl', 'wb') as file_cmc:
            pickle.dump(model_complete, file_cmc)
        with open('Outputs\csp_model_reducted_train.pkl', 'wb') as file_cmr:
            pickle.dump(model_reducted, file_cmr)
        with open('Outputs\csp_model_rows_root_train.pkl', 'wb') as file_cmrr:
            pickle.dump(model_rows_root, file_cmrr)
    elif mode == "full":
        with open('Outputs\csp_model_complete_full.pkl', 'wb') as file_cmc:
            pickle.dump(model_complete, file_cmc)
        with open('Outputs\csp_model_reducted_full.pkl', 'wb') as file_cmr:
            pickle.dump(model_reducted, file_cmr) 
        with open('Outputs\csp_model_rows_root_full.pkl', 'wb') as file_cmrr:
            pickle.dump(model_rows_root, file_cmrr)
    # ------------------------------



# Function to ask the csp() models, previously saved in files.
# Input rows : rows to be predicted (without the prediction column).
# Input mode : "train" or "full" so that the function works on the full model or on the train model.
# Output : predictions associated to each input row.
def ask_csp(rows, mode):
    
    # ------------------------------
    # Load models from files
    model_complete = {}
    model_reducted = {}
    model_rows_root = {}
    if mode == "train":
        with open('Outputs\csp_model_complete_train.pkl', 'rb') as file_cmc:
            model_complete = pickle.load(file_cmc) 
        with open('Outputs\csp_model_reducted_train.pkl', 'rb') as file_cmr:
            model_reducted = pickle.load(file_cmr)
        with open('Outputs\csp_model_rows_root_train.pkl', 'rb') as file_cmrr:
            model_rows_root = pickle.load(file_cmrr) 
    elif mode == "full":
        with open('Outputs\csp_model_complete_full.pkl', 'rb') as file_cmc:
            model_complete = pickle.load(file_cmc)
        with open('Outputs\csp_model_reducted_full.pkl', 'rb') as file_cmr:
            model_reducted = pickle.load(file_cmr)
        with open('Outputs\csp_model_rows_root_full.pkl', 'rb') as file_cmrr:
            model_rows_root = pickle.load(file_cmrr)
    # ------------------------------
    
    # We will get predictions using mrr on one hand and mc_mr on the other hand.
    # We will compare predictions between mrr and mc_mr (respectively stored in results_mrr and results_mc_mr). Depending on the comparison we will add final predictions to results.
    results = []
    results_mrr = []
    results_mc_mr = []
    
    # For each row to predict.
    for row in range(0, len(rows)):
        # Prints
        print(LINE_UP, end=LINE_CLEAR)
        print("Asking For Row " + colored(row+1, 'yellow') + "/" + str(len(rows)))
        # Get the row to predict
        to_predict = rows[row]
        s_row_split = to_predict
        # The array to save predicted values having the same max val
        pvs_from_mrr = []
        
        
        # ------------------------------
        # Ask model_rows_root (mrr)
        # ------------------------------  
        
        # Calculation of similarity percentages between row and model_rows_root rows. 
        # We keep the model rows having the best similarity percentage above or equal to posim%.
        # The mrr results are then determined according to these mrr rows.
        posim = 0.8
        sim_results = {}
        best_sim_score = 0
        # Compare the row to predict to each row in mrr in order to determine the similarity score
        for mrr_row in model_rows_root:
            # Prepare the row.
            mrr_row_split = mrr_row.split("@@@")
            mrr_row_split.pop()
            # Calculate similarity score.
            sim_score = 0
            for i in range (0, len(s_row_split)):
                if s_row_split[i] == mrr_row_split[i]:
                    sim_score += 1/len(s_row_split)
            sim_score = round(sim_score, 2)
            # If the similarity score is >= posim and >= best similarity score, save the mrr_row in sim_results. Update the best similarity score if needed.
            if sim_score >= posim:
                if sim_score > best_sim_score:
                    # Update the best similarity score.
                    best_sim_score = sim_score
                    # Reset sim_results.
                    sim_results = {}
                    # Add the row to sim_results.
                    new_key_value = {mrr_row:sim_score}
                    sim_results.update(new_key_value)
                elif sim_score == best_sim_score:
                    # Add the row to sim_results.
                    new_key_value = {mrr_row:sim_score}
                    sim_results.update(new_key_value)
                
        # If at least one similar row has been found in model_rows_root.
        if len(sim_results) >= 1:
            # Agregate all sim_results pvs by summing their associated values.
            pvs_fused = {}
            for sim_row in sim_results:
                for key in model_rows_root[sim_row]:
                    value = model_rows_root[sim_row][key]
                    if key not in pvs_fused:
                        new_key_value = {key:value}
                        pvs_fused.update(new_key_value)
                    else:
                        pvs_fused[key] += value
            # If there is only one pv in pvs_fused, this pv is the final pv predicted for the row
            if len(pvs_fused) == 1:
                predicted_value = list(pvs_fused.keys())[0]
                results_mrr.append([predicted_value])
            # Otherwise, a decision must be made between several pvs, looking to see if several pvs share the maximum value.
            else:
                # Taking list of values in pvs_fused
                v = list(pvs_fused.values())
                # Taking list of keys in pvs_fused
                k = list(pvs_fused.keys())
                # Save max value from v and its index
                max_val = max(v)
                max_val_index = v.index(max(v))
                # Delete max value from v
                del v[max_val_index]
                # If new max val from v is equal to the deleted one, it means that the max value in v was multiple.
                # If max_val is not multiple, the associated pv is the final predicted pv for the row.
                if max(v) < max_val:
                    # predicted_value is the key having the unique max value
                    predicted_value = k[max_val_index]
                    results_mrr.append([predicted_value])
                # Else, max val from v is multiple so we save all pvs having the max val in pvs_from_mrr.
                else:
                    # Save the first pv discovered having the max val
                    pvs_from_mrr.append(k[max_val_index])
                    del k[max_val_index]
                    # As long as equal max val values are found, they are stored in pvs_from_mrr
                    while len(v)>0 and max(v) == max_val:
                        max_val_index = v.index(max(v))
                        pvs_from_mrr.append(k[max_val_index])
                        del v[max_val_index]
                        del k[max_val_index]
                    results_mrr.append(pvs_from_mrr)
        # Else, no similar row has been found in model_rows_root so we append an empty array in mrr results. It will be used for the comparison step with mc_mr results.
        else:
            results_mrr.append([])
            
            
        # --------------------------------------
        # Ask model_complete and model_reducted
        # --------------------------------------
        
        # To query mc and mr, i.e. to find the prediction scores (sp) associated with each predicted_value for the current key "question" (row), we need to calculate, for each predicted_value, the following prediction scores (sp).
        
        # Raw sp with full model.
        # Raw similarity percentage between the "question" key and the predicted_value key, without considering the scores associated with the X values.
        scores_brut_mc = {}
        for pv in model_complete:
            score = 0
            for i in range(0, len(to_predict)):
                if to_predict[i] in model_complete[pv][i]:
                    score += 1
            score = score/len(to_predict)
            new_key_value = {pv:score}
            scores_brut_mc.update(new_key_value)
        
        # Refined sp with full model.
        # Average of the scores of the X values in correspondence between the "question" key and the predicted_value key.
        scores_fine_mc = {}
        for pv in model_complete:
            score = 0
            for i in range(0, len(to_predict)):
                if to_predict[i] in model_complete[pv][i]:
                    score += model_complete[pv][i][to_predict[i]]
            score = score/len(to_predict)
            new_key_value = {pv:score}
            scores_fine_mc.update(new_key_value)
        
        # Raw sp with reduced model.
        # Raw similarity percentage between the "question" key and the predicted_value key, without considering the scores associated with the X values.          
        scores_brut_mr = {}
        for pv in model_reducted:
            score = 0
            for i in range(0, len(to_predict)):
                if to_predict[i] in model_reducted[pv][i]:
                    score += 1
            score = score/len(to_predict)
            new_key_value = {pv:score}
            scores_brut_mr.update(new_key_value)
        
        # Refined sp with reduced model.
        # Average of the scores of the X values in correspondence between the "question" key and the predicted_value key.
        scores_fine_mr = {}
        for pv in model_reducted:
            score = 0
            for i in range(0, len(to_predict)):
                if to_predict[i] in model_reducted[pv][i]:
                    score += model_reducted[pv][i][to_predict[i]]
            score = score/len(to_predict)
            new_key_value = {pv:score}
            scores_fine_mr.update(new_key_value)
        
        # ----------------------------------
        # Final predicted_value scores
        # ----------------------------------
        # sbmc = scores_brut_mc
        # sfmc = scores_fine_mc
        # sbmr = scores_brut_mr
        # sfmr = scores_fine_mr
        scores_final = {}
        for pv in model_complete: 
            # --------------------
            
            # Calculation method : Mean(sbmc, sfmc, sbmr, sfmr).
            score = mean([
                            scores_brut_mc[pv],
                            scores_fine_mc[pv],
                            scores_brut_mr[pv],
                            scores_fine_mr[pv]
            ])
            new_key_value = {pv:score}
            scores_final.update(new_key_value)
            
            # --------------------
            """
            # Calculation method : Mean(sbmc, sfmc, sbmr) with coefficient on sbmc.
            # => TOP
            # => A high coefficient on sbmc increases accuracy by a few %.
            coef = 1
            score = mean([
                            coef*scores_brut_mc[pv],
                            scores_fine_mc[pv],
                            scores_brut_mr[pv]
                        ])
            new_key_value = {pv:score}
            scores_final.update(new_key_value)
            """
            # --------------------
            """
            # Calculation method : Mean(sfmc, sbmc)
            score = mean([
                            scores_fine_mc[pv],
                            scores_brut_mc[pv]
                        ])
            new_key_value = {pv:score}
            scores_final.update(new_key_value)
            """
            # --------------------
            """
            # Calculation method : sfmc only
            score = scores_fine_mc[pv]
            new_key_value = {pv:score}
            scores_final.update(new_key_value)
            """
            # --------------------

        """
        # Get the pv having the best score in scores_final and save it in results.
        predicted_value = max(scores_final, key=scores_final.get)
        results_mc_mr.append(predicted_value)
        """
        
        # Retrieve some best pvs from scores_final, sorted in DESC order.
        sorted_sf = sorted(scores_final.items(), key = lambda x:x[1], reverse = True)
        bests = [sorted_sf[0][0], sorted_sf[1][0], sorted_sf[2][0]]
        results_mc_mr.append(bests)
        
    
    # Compare results_mrr and results_mc_mr to build final results.
    for i in range(0, len(results_mc_mr)):
        final_result_found = False
        for x in range(0, len(results_mc_mr[i])):
            if final_result_found == False:
                # If the currently treated mc_mr result is in mrr results, it is the final result so it is added to results.
                if results_mc_mr[i][x] in results_mrr[i]:
                    results.append(results_mc_mr[i][x])
                    final_result_found = True
        # If no match was found in previous step.
        if final_result_found == False:
            # If there are no entries in mrr results, the final result is considered to be the first result from mc_mr. This result is added to results.
            if len(results_mrr[i]) == 0:
                results.append(results_mc_mr[i][0])
            # Else, the final result is a random result from either mrr results or all results from mrr and mc_mr.
            else:
                # --------------------
                # The final result is a random result from all results from mrr and mc_mr. This result is added to results.
                # fused_results = results_mrr[i]
                # fused_results.append(results_mc_mr[i][0])
                # rand = random.randint(0, len(fused_results)-1)
                # results.append(fused_results[rand])
                # --------------------
                # The final result is a random result from mrr results. This result is added to results.
                rand = random.randint(0, len(results_mrr[i])-1)
                results.append(results_mrr[i][rand])
                # --------------------
    
    print(LINE_UP, end=LINE_CLEAR)
    
    return results

    
# Call ask_csp() directly from the console.
# ask_csp() will be called using the file given as parameter without the .txt extension.
# The file must be placed in the folder named "Data" and must contain lines of values separated by commas and without the prediction column, which means only the X predictor variables.
# Line Example : 0,F,2,7,1501-2500,1,oui,0-1,4,0-1,6,month-03,jeudi,1
# Call command : py 4_functions.py ask_csp file_name
import sys
import time
from icecream import ic
import pickle
import csv
import itertools
import threading
from termcolor import colored
from statistics import mean

# Line up and clear for prints
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

if sys.argv[1] == "ask_csp":
    
    # Get 2nd parameter. Name of the file containing rows to predict.
    try:
        filename = sys.argv[2]
    except:
        print("You must specify a file name as 2nd parameter, without the .txt extension. The file must contain lines of values separated by commas and without the prediction column.")
        sys.exit()
    
    print("Asking csp...")
    
    # Load file as a matrix.
    # The file must be placed in the folder named "Data" and must contain lines of values separated by commas and without the prediction column.
    rows = []
    with open('Data\\'+filename+'.txt', newline='') as csvfile:
        rows = list(csv.reader(csvfile))
    
    # Call ask_csp().
    results = ask_csp(rows, "full")
    
    # Display results.
    print("------------------------")
    print(colored("RESULTS", 'yellow'))
    print("------------------------")
    ic.configureOutput(prefix="Ask for : ")
    for i in range(0, len(results)):
        ic(str(rows[i]))
        print("Predicted Value : " + colored(results[i], 'yellow'))
        print('------------------------------')



# Function to evaluate a ML model using various metrics.
# y_test : the array of correct classes associated to X_test.
# y_pred : the array of predicted classes, from the model, for X_test.
def evaluate_model(y_test, y_pred):
    
    evaluation_metrics = {}
    
    # accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    # print(LINE_UP, end=LINE_CLEAR)
    print("Accuracy: " + colored(str(round(accuracy*100)), 'yellow') + "%")
    new_key_value = {"accuracy":accuracy}
    evaluation_metrics.update(new_key_value)
    
    # balanced_accuracy_score
    # The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
    # The best value is 1 and the worst value is 0 when adjusted=False.
    # balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    # precision_score
    # The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    # The best value is 1 and the worst value is 0.
    # The average parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
    # 'binary':
    # Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    # 'micro':
    # Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro':
    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted':
    # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    # 'samples':
    # Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    new_key_value = {"precision_micro":precision_micro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"precision_macro":precision_macro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"precision_weighted":precision_weighted}
    evaluation_metrics.update(new_key_value)
        
    # f1_score
    # Compute the F1 score, also known as balanced F-score or F-measure.
    # The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
    # F1 = 2 * (precision * recall) / (precision + recall)
    # In the multi-class and multi-label case, this is the average of the F1 score of each class with weighting depending on the average parameter.
    # The average parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:
    # 'binary':
    # Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    # 'micro':
    # Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro':
    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted':
    # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    # 'samples':
    # Calculate metrics for each instance, and find their average (only meaningful for multilabel classification where this differs from accuracy_score).
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    new_key_value = {"f1_micro":f1_micro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"f1_macro":f1_macro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"f1_weighted":f1_weighted}
    evaluation_metrics.update(new_key_value)
    
    # fbeta_score
    # The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.
    # The beta parameter represents the ratio of recall importance to precision importance. beta > 1 gives more weight to recall, while beta < 1 favors precision. For example, beta = 2 makes recall twice as important as precision, while beta = 0.5 does the opposite. Asymptotically, beta -> +inf considers only recall, and beta -> 0 only precision.
    fbeta_micro = fbeta_score(y_test, y_pred, average='micro', beta=0.5, zero_division=0)
    fbeta_macro = fbeta_score(y_test, y_pred, average='macro', beta=0.5, zero_division=0)
    fbeta_weighted = fbeta_score(y_test, y_pred, average='weighted', beta=0.5, zero_division=0)
    new_key_value = {"fbeta_micro":fbeta_micro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"fbeta_macro":fbeta_macro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"fbeta_weighted":fbeta_weighted}
    evaluation_metrics.update(new_key_value)
    
    # recall_score
    # The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
    # The best value is 1 and the worst value is 0.    
    recall_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    new_key_value = {"recall_micro":recall_micro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"recall_macro":recall_macro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"recall_weighted":recall_weighted}
    evaluation_metrics.update(new_key_value)
    
    # log_loss
    # Log loss, aka logistic loss or cross-entropy loss.
    # This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks, defined as the negative log-likelihood of a logistic model that returns y_pred probabilities for its training data y_true. The log loss is only defined for two or more labels.
    
    # roc_auc_score
    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    # Note: this implementation can be used with binary, multiclass and multilabel classification, but some restrictions apply (see Parameters).
       
    # zero_one_loss
    
    # brier_score_loss
    
    # jaccard_score
    # Jaccard similarity coefficient score.
    # The Jaccard index [1], or Jaccard similarity coefficient, defined as the size of the intersection divided by the size of the union of two label sets, is used to compare set of predicted labels for a sample to the corresponding set of labels in y_true.    
    jaccard_micro = jaccard_score(y_test, y_pred, average='micro', zero_division=0)
    jaccard_macro = jaccard_score(y_test, y_pred, average='macro', zero_division=0)
    jaccard_weighted = jaccard_score(y_test, y_pred, average='weighted', zero_division=0)
    new_key_value = {"jaccard_micro":jaccard_micro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"jaccard_macro":jaccard_macro}
    evaluation_metrics.update(new_key_value)
    new_key_value = {"jaccard_weighted":jaccard_weighted}
    evaluation_metrics.update(new_key_value)
    
    # cohen_kappa_score
    
    # hinge_loss
    
    # matthews_corrcoef
    # Compute the Matthews correlation coefficient (MCC).
    # The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient. [source: Wikipedia]
    # Binary and multiclass labels are supported. Only in the binary case does this relate to information about true and false positives and negatives.    
    matthews_cc = matthews_corrcoef(y_test, y_pred)
    new_key_value = {"matthews_cc":matthews_cc}
    evaluation_metrics.update(new_key_value)
      
    # hamming_loss
    # Compute the average Hamming loss.
    # The Hamming loss is the fraction of labels that are incorrectly predicted.    
    # In multiclass classification, the Hamming loss corresponds to the Hamming distance between y_true and y_pred which is equivalent to the subset zero_one_loss function, when normalize parameter is set to True.
    # In multilabel classification, the Hamming loss is different from the subset zero-one loss. The zero-one loss considers the entire set of labels for a given sample incorrect if it does not entirely match the true set of labels. Hamming loss is more forgiving in that it penalizes only the individual labels.
    # The Hamming loss is upperbounded by the subset zero-one loss, when normalize parameter is set to True. It is always between 0 and 1, lower being better.    
    ham_loss = hamming_loss(y_test, y_pred)
    new_key_value = {"ham_loss":ham_loss}
    evaluation_metrics.update(new_key_value)
    
    # confusion_matrix
    
    # multilabel_confusion_matrix
    
    # precision_recall_curve
    
    # roc_curve 

    return evaluation_metrics


def naive_bayes():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str')
    
    # Encode the categorical features
    encoder = LabelEncoder()
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.005, random_state=None
    )

    # Create and train model
    model = CategoricalNB()
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    # Use model
    # Question row example : [3,1,0]
    # print("\nPredicting class...")
    # probs = model.predict_proba([[3,1,0]])
    # print("\nPrediction probs: ")
    # print(probs)
    # predicted = model.predict([[3,1,0]])
    # print("\nPredicted class: ")
    # print(predicted)
    
    # Save model using pickle
    # import pickle
    # print("Saving trained naive Bayes model ")
    # path = ".\\Models\\bayes_scikit_model.sav"
    # pickle.dump(model, open(path, "wb"))
    
    # Predict by loading the model
    # x = np.array([[1, 0, 2]], dtype=np.int64)
    # with open(path, 'rb') as f:
    #   loaded_model = pickle.load(f)
    # pa = loaded_model.predict_proba(x)
    # print(pa)    
    
    return [
        evaluation_metrics
    ]


def decision_tree():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str') 
    
    # Encode the categorical features
    encoder = LabelEncoder()
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.005, random_state=None
    )

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    # --------------------
    """
    # Optimized Decision Tree
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(round(accuracy*100)) + "%")
    """
    
    return [
        evaluation_metrics
    ]

    
def logistic_regression():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str')
    
    # Encode the categorical features
    encoder = LabelEncoder()
    # Encode X
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    # Encode y
    # y = encoder.fit_transform(y)
    
    # Convert X to numeric values
    X = X.astype(int)
    # Convert y to numeric values
    # y = y.astype(int)
    
    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.005, random_state=None
    )
    
    # Instantiate the model (using the default parameters)
    # Possible solvers : 'newton-cholesky', 'lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'
    logreg = LogisticRegression(
        random_state=None, 
        solver='newton-cholesky', 
        max_iter=100
    )

    # Fit the model with data
    logreg.fit(X_train, y_train)

    # Predict on test data
    y_pred = logreg.predict(X_test)

    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    return [
        evaluation_metrics
    ]
    

def linear_regression():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str')
    
    # Encode the categorical features
    encoder = LabelEncoder()
    # Encode X
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    # Encode y
    y = encoder.fit_transform(y)
    
    # Convert X to numeric values
    X = X.astype(int)
    # Convert y to numeric values
    y = y.astype(int)

    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.005, random_state=None
    )

    # Model
    lm = LinearRegression()
    
    # Fit the model with data
    lm.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = lm.predict(X_test)
    
    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    return [
        evaluation_metrics
    ]
    

def neural_network():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str')
    
    # Encode the categorical features
    encoder = LabelEncoder()
    # Encode X
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    # Encode y
    y = encoder.fit_transform(y)
    
    # Convert X to numeric values
    X = X.astype(int)
    # Convert y to numeric values
    y = y.astype(int)
    
    """
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X).reshape((50,5))
    """
    """
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()
    """

    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.005, random_state=None
    )
    
    # Model
    """
    creates a Sequential model to add layers one at a time.
    the first layer expects four arguments:
        8: No.of.neurons present in that layer.
        input_dim: specify the dimension of the input data.
        init: specify whether uniform or normal distribution of weights to be initialized.
        activation: specify whether relu or sigmoid or tanh activation function to be used for each neuron in that layer.
    the next hidden layer has 6 neurons with an uniform initialization of weights and relu activation function
    the output layer has only one neuron as this is a binary classification problem. The activation function at output is sigmoid because it outputs a probability in the range 0 and 1 so that we could easily discriminate output by assigning a threshold.
    """
    model = Sequential()
    model.add(Dense(8, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    """
    After creating the model, three parameters are needed to compile the model in Keras.

    loss: This is used to evaluate a set of weights. It is needed to reduce the error between actual output and expected output. It could be binary_crossentropy or categorical_crossentropy depending on the problem. As we are dealing with a binary classification problem, we need to pick binary_crossentropy. Here is the list of loss functions available in Keras.
    optimizer: This is used to search through different weights for the network. It could be adam or rmsprop depending on the problem. Here is the list of optimizers available in Keras.
    metrics: This is used to collect the report during training. Normally, we pick accuracy as our performance metric. Here is the list of metrics available in Keras.
    These parameters are to be tuned according to the problem as our model needs some optimization in the background (which is taken care by Theano or TensorFlow) so that it learns from the data during each epoch (which means reducing the error between actual output and predicted output).

    epoch is the term used to denote the number of iterations involved during the training process of a neural network.

    chooses a binary_crossentropy loss function and the famous Stochastic Gradient Descent (SGD) optimizer adam. It also collects the accuracy metric for training.
    """
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    """
    After compiling the model, the dataset must be fitted with the model. The fit() function in Keras expects five arguments -

    X_train: the input training data.
    Y_train: the output training classes.
    validation_data: Tuple of testing or validation data used to check the performance of our network.
    nb_epoch: how much iterations should the training process take place.
    batch_size: No.of.instances that are evaluated before performing a weight update in the network.

    chooses 100 iterations to be performed by the deep neural network with a batch_size of 5.
    """
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=5)

    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    return [
        evaluation_metrics
    ]


def neural_net_categorical():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str')

    # Encode the categorical features
    encoder = LabelEncoder()
    # Encode X
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    # Encode y
    y = encoder.fit_transform(y)
    
    # Convert X to numeric values
    X = X.astype(int)
    # Convert y to numeric values
    y = y.astype(int)
    
    # Convert y to categorical.
    # Example : You have three classes: "Cat", "Dog", "Fish", represented by class indices 0, 1, and 2 respectively. If you have an array of labels (y) [1, 0, 2, 1], the one-hot encoding would look like:
    """
    [0, 1, 0]  # Dog
    [1, 0, 0]  # Cat
    [0, 0, 1]  # Fish
    [0, 1, 0]  # Dog
    """
    y = to_categorical(y) 
    
    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.005, random_state=None
    )
    
    # Get the number of features (number of columns in X) and the number of classes (number of distinct values in y).
    num_features = len(X_train[0])
    num_classes = len(y[0])

    # Model    
    # Deep Neural Network
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

    # Fit
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Convert y_pred to Class Labels.
    # The model is outputting class probabilities for each example in X_test. So we need to convert these probabilities to class labels by taking the index of the maximum probability for each example. This is often done using the numpy.argmax function.
    y_pred_labels = np.argmax(y_pred, axis=1)
    # We also need to reconvert y_test to class labels.
    y_test_labels = np.argmax(y_test, axis=1)
    # We can now compare these two arrays to evaluate the model.
    
    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test_labels, y_pred_labels)
    
    return [
        evaluation_metrics
    ]


def k_nearest_neighbours():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str')

    # Encode the categorical features
    encoder = LabelEncoder()
    # Encode X
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    # Encode y
    # y = encoder.fit_transform(y)
    
    # Convert X to numeric values
    X = X.astype(int)
    # Convert y to numeric values
    # y = y.astype(int)
    
    # Splitting data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=None)

    """
    Before diving further, let’s take a look at the KNeighborsClassifier class:

    KNeighborsClassifier(
        n_neighbors=5,          # The number of neighbours to consider
        weights='uniform',      # How to weight distances
        algorithm='auto',       # Algorithm to compute the neighbours
        leaf_size=30,           # The leaf size to speed up searches
        p=2,                    # The power parameter for the       
                                  Minkowski metric
        metric='minkowski',     # The type of distance to use
        metric_params=None,     # Keyword arguments for the metric 
                                  function
        n_jobs=None             # How many parallel jobs to run
    )
    """

    # Creating a classifier object in sklearn
    clf = KNeighborsClassifier(p=1)

    # Fit
    clf.fit(X_train, y_train)

    # Making predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    return [
        evaluation_metrics
    ]
    

def k_means():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str')

    # Encode the categorical features
    encoder = LabelEncoder()
    # Encode X
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    # Encode y
    y = encoder.fit_transform(y)
    
    # Convert X to numeric values
    X = X.astype(int)
    # Convert y to numeric values
    y = y.astype(int)
    
    # Get the number of classes (number of distinct values in y).
    distinct_y = set(y)
    num_classes = len(distinct_y) 
    
    # Splitting data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=None)
    
    # Creating a classifier object in sklearn.
    # The parameter corresponds to the number of classes in y.
    model = KMeans(
                    init="random",
                    n_clusters=num_classes,
                    n_init="auto",
                    max_iter=500,
                    random_state=None
                )

    # Fit
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    return [
        evaluation_metrics
    ]
    

def support_vector_machine():
    # Load data
    file_path = "Outputs/df.csv"
    # Determining the number of columns in the dataset.
    with open(file_path) as f:
        n_cols = len(f.readline().split(","))
    # Load
    X = np.loadtxt(file_path, usecols=range(0,n_cols-2), delimiter=",", dtype='str')
    y = np.loadtxt(file_path, usecols=n_cols-1, delimiter=",", dtype='str')

    # Encode the categorical features
    encoder = LabelEncoder()
    # Encode X
    X = np.transpose(X)
    for row in range(0, len(X)):
        X[row] = encoder.fit_transform(X[row])
    X = np.transpose(X)
    # Encode y
    # y = encoder.fit_transform(y)
    
    # Convert X to numeric values
    X = X.astype(int)
    # Convert y to numeric values
    # y = y.astype(int)
    
    # Splitting data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005, random_state=None)    
    
    # Create a svm Classifier
    # Kernels : "linear", "poly", "rbf"
    # The "poly" and "rbf" kernels are especially useful when the data-points are not linearly separable.
    clf = svm.SVC(kernel='rbf')

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Evaluate the model
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    return [
        evaluation_metrics
    ]
    





# OLD

    """
    # ------------------------------
    # Adjust the model
    print(LINE_UP, end=LINE_CLEAR)
    print("Adjusting Model...")
    
    # ROLLBACK: Save models, used for rollbacks
    old_mc = deepcopy(model_complete)
    old_mr = deepcopy(model_reducted)

    # Getting real y values
    y_real = []
    for tab in X_train:
        y_real.append(tab[-1])
        del tab[-1] 
        
    # Compute accuracy on full X_train
    best_accuracy = 0
    y_pred = ask_csp(X_train, mode, model_complete, model_reducted)
    for i in range(0, len(y_real)):
        if y_real[i] == y_pred[i]:
            best_accuracy += 1/len(y_real)
            
    print("Initial Accuracy: " + str(round(best_accuracy*100)) + "%")
    print("")
    
    # Do adjustment loops while accuracy does not stagnate on too many iterations.
    max_iteration_stagnation = 50
    stagnation_counter = 0
    loop_num = 0
    while stagnation_counter <= max_iteration_stagnation:
        loop_num += 1
        
        # Extract p% random rows from X_train (called X_train_extract). Build y_real_extract in accordance.
        p = 0.01
        X_train_extract = []
        y_real_extract = []
        rows_qty_to_extract = round(len(X_train) * p)
        for i in range(0, rows_qty_to_extract):
            random_index = random.randint(0, len(X_train)-1)
            X_train_extract.append(X_train[random_index])
            y_real_extract.append(y_real[random_index])
                
        for row_ext in range(0, len(X_train_extract)-1):
            
            # If the prediction is wrong
            predicted_value = ask_csp(
                                    [X_train_extract[row_ext]], 
                                    mode, 
                                    model_complete,
                                    model_reducted
                                )
            if y_real_extract[row_ext] != predicted_value[0]:

                # Boost values of X variables corresponding to pv y_real_extract[row_ext] in model_complete.
                # Unboost values of X variables corresponding to pv y_pred[row_ext] in model_complete.
                key_to_boost = deepcopy(y_real_extract[row_ext])
                key_to_unboost = deepcopy(predicted_value[0])
                
                # Calculating the weights used to boost or unboost values in the model. It uses the same formula as the one used to make the model, see above (variable percent_value).
                # weight for the key to boost
                n_pv_rows_boost = len(df_ex[key_to_boost])
                weight_boost = (1/n_pv_rows_boost) + (1/X_train_rows) + (n_pv_rows_boost/X_train_rows**2)
                # weight for the key to unboost
                n_pv_rows_unboost = len(df_ex[key_to_unboost])
                weight_unboost = (1/n_pv_rows_unboost) + (1/X_train_rows) + (n_pv_rows_unboost/X_train_rows**2)
                
                for x in range(0, len(X_train_extract[row_ext])):
                    
                    x_val = deepcopy(X_train_extract[row_ext][x])
                    
                    # Boost
                    weight = weight_boost
                    model_complete[key_to_boost][x][x_val] += weight
                    
                    # Unboost
                    weight = weight_unboost
                    if x_val in model_complete[key_to_unboost][x]:
                        model_complete[key_to_unboost][x][x_val] -= weight
                    else:
                        # Ajout de la valeur x_val
                        new_key_value = {x_val:-weight}
                        model_complete[key_to_unboost][x].update(new_key_value)
                        
                # ------------------------------
                # Calculate the reduced model by removing from the full model all values with a presence percentage < reduc_value
                model_reducted = deepcopy(model_complete)
                for key in model_complete:
                    for column in model_complete[key]:
                        for value in model_complete[key][column]:
                            if model_complete[key][column][value] < reduc_value:
                                del model_reducted[key][column][value]
        
        # Compute loop final accuracy on X_train
        y_pred = ask_csp(X_train, mode, model_complete, model_reducted)
        final_accuracy = 0
        for i in range(0, len(y_real)):
            if y_real[i] == y_pred[i]:
                final_accuracy += 1/len(y_real)
        
        # If an improvment occurred.        
        if final_accuracy > best_accuracy:
            # Reset stagnation_counter
            stagnation_counter = 0
            # ROLLBACK: Save models for future rollbacks
            old_mc = deepcopy(model_complete)
            old_mr = deepcopy(model_reducted)
            # Update best_accuracy
            best_accuracy = final_accuracy
        elif final_accuracy == best_accuracy:
            # ROLLBACK: Save models for future rollbacks
            old_mc = deepcopy(model_complete)
            old_mr = deepcopy(model_reducted)
        else:
            # Increment stagnation_counter
            stagnation_counter += 1
            # ROLLBACK: Put models to former better versions
            model_complete = deepcopy(old_mc)
            model_reducted = deepcopy(old_mr)
        
        # Print infos
        print(LINE_UP, end=LINE_CLEAR)
        print("Adjusting - Loop " + str(colored(loop_num, 'yellow')) + " - Stagnation " + str(colored(stagnation_counter,'yellow')) + "/" + str(max_iteration_stagnation) + " - Accuracy: " + colored(round(best_accuracy*100, 2), 'yellow') + "%")
    
    print(LINE_UP, end=LINE_CLEAR)   
    print("Adjustment Done: " + colored(str(loop_num), 'yellow') + " Loops - Accuracy: " + colored(round(best_accuracy*100, 2), 'yellow') + "%")

    # End of model adjustment
    # ------------------------------
    """
    
    """
    # ------------------------------
    # Adjust the model
    print(LINE_UP, end=LINE_CLEAR)
    print("Adjusting Model...")
    
    # ROLLBACK: Save old_model, used for rollbacks
    old_model = deepcopy(model_complete)

    # Getting real y values
    y_real = []
    for tab in X_train:
        y_real.append(tab[-1])
        del tab[-1] 
    
    # Do adjustment loops while accuracy does not stagnate on too many iterations.
    max_iteration_stagnation = 50
    stagnation_counter = 0
    loop_num = 0
    while stagnation_counter <= max_iteration_stagnation:
        loop_num += 1
        
        # Extract a test set from X_train (called X_train_test). Build y_real_test.
        p = 0.1
        X_train_test = []
        y_real_test = []
        rows_qty_to_extract = round(len(X_train) * p)
        for i in range(0, rows_qty_to_extract):
            random_index = random.randint(0, len(X_train)-1)
            X_train_test.append(X_train[random_index])
            y_real_test.append(y_real[random_index])
             
        # Compute start accuracy on X_train_test.
        # Store badly predicted rows and respective real y values and respective predicted values in "X_bad" and "y_bad_real" and "y_bad_pred"
        start_accuracy = 0
        X_bad = []
        y_bad_real = []
        y_bad_pred = []
        y_pred = ask_csp(X_train_test, mode, model_complete, model_reducted)
        for i in range(0, len(y_real_test)):
            if y_real_test[i] == y_pred[i]:
                start_accuracy += 1/len(y_real_test)
            else:
                X_bad.append(X_train_test[i])
                y_bad_real.append(y_real_test[i])
                y_bad_pred.append(y_pred[i])
             
        # Modify all badly predicted rows
        for row_ext in range(0, len(X_bad)-1):

            updated_pred = ask_csp([X_bad[row_ext]], mode, model_complete, model_reducted)
            
            if updated_pred[0] != y_bad_real[row_ext]:
                # Boost values of X variables corresponding to pv y_real_extract[row_ext] in model_complete.
                # Unboost the values of X variables corresponding to the y_pred[row_ext] pv in model_complete.
                key_to_boost = deepcopy(y_bad_real[row_ext])
                key_to_unboost = deepcopy(y_bad_pred[row_ext])
                
                # Calculating the weights used to boost or unboost values in the model. It uses the same formula as the one used to make the model, see above (variable percent_value).
                # weight for the key to boost
                n_pv_rows_boost = len(df_ex[key_to_boost])
                weight_boost = (1/n_pv_rows_boost) + (1/X_train_rows) + (n_pv_rows_boost/X_train_rows**2)
                # weight for the key to unboost
                n_pv_rows_unboost = len(df_ex[key_to_unboost])
                weight_unboost = (1/n_pv_rows_unboost) + (1/X_train_rows) + (n_pv_rows_unboost/X_train_rows**2)
                
                for x in range(0, len(X_bad[row_ext])):
                    
                    x_val = deepcopy(X_bad[row_ext][x])
                    
                    # Boost
                    weight = weight_boost
                    model_complete[key_to_boost][x][x_val] += weight
                    # Unboost
                    weight = weight_unboost
                    if x_val in model_complete[key_to_unboost][x]:
                        model_complete[key_to_unboost][x][x_val] -= weight
                    else:
                        # Add x_val value
                        new_key_value = {x_val:-weight}
                        model_complete[key_to_unboost][x].update(new_key_value)
                
        # Compute final accuracy on X_train_test.
        y_pred = ask_csp(X_train_test, mode, model_complete, model_reducted)
        final_accuracy = 0
        for i in range(0, len(y_real_test)):
            if y_real_test[i] == y_pred[i]:
                final_accuracy += 1/len(y_real_test)
        
        # If an improvment occurred.        
        if final_accuracy > start_accuracy:
            # Reset stagnation_counter
            stagnation_counter = 0
            # ROLLBACK: Save old_model for future rollbacks
            old_model = deepcopy(model_complete)
        elif final_accuracy == start_accuracy:
            # ROLLBACK: Save old_model for future rollbacks
            old_model = deepcopy(model_complete)
        else:
            # Increment stagnation_counter
            stagnation_counter += 1
            # ROLLBACK: Put model_complete to former better version
            model_complete = deepcopy(old_model)
        
        # Print infos
        diff_acc = final_accuracy - start_accuracy
        print(LINE_UP, end=LINE_CLEAR)
        print("Adjusting - Loop " + str(colored(loop_num, 'yellow')) + " - Stagnation " + str(colored(stagnation_counter,'yellow')) + "/" + str(max_iteration_stagnation) + " - Diff Acc: " + colored(round(diff_acc*100), 'yellow') + "%")
    
    print(LINE_UP, end=LINE_CLEAR)   
    print("Adjustment Done: " + colored(str(loop_num), 'yellow') + " Loops")

    # End of model adjustment
    # ------------------------------
    """
    
    """
    # ------------------------------
    # Calculate the reduced model by removing from the full model all values with a presence percentage < reduc_value
    reduc_value = 0.1
    model_reducted = deepcopy(model_complete)
    for key in model_complete:
        for column in model_complete[key]:
            for value in model_complete[key][column]:
                if model_complete[key][column][value] < reduc_value:
                    del model_reducted[key][column][value]
    """
    