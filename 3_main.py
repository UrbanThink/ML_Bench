# Imports
from past.builtins import execfile
execfile('1_imports.py')
execfile('4_functions.py')
# exec(open("./filename").read()) # => for Unix machines

# Line up and clear for prints
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# Get parameters/Global_Variables.
# Reminder : In Python, accessing global variables inside functions is ok but if we want to MODIFY a global variable inside a function, we must declare it as global inside the function or use a singleton pattern.

# Get df.
df = []
with open(sys.argv[1], newline='') as csvfile:
  df = list(csv.reader(csvfile))
  
# Macro iterations.
macro_iterations = int(sys.argv[2])

# Algorithms to execute. Also used for files saved.
# Available algorithms : "naive_bayes", "decision_tree", "logistic_regression", "neural_net_categorical", "k_nearest_neighbours", "support_vector_machine", "csp"
# Not suitable for categorical predictions : "linear_regression", "neural_network", "k_means"
algo_names = (sys.argv[3]).split("-")

# Currently executed algorithm
current_algo = ""


# EXCEL FILE
# Workbook() argument is the filename to create.
workbook = xlsxwriter.Workbook('Outputs/RESULTS.xlsx') 
# The workbook object is then used to add new worksheet(s).
worksheet = workbook.add_worksheet()
worksheet.write('A1', "Executions = "+str(macro_iterations)) 
worksheet.write('B1', "Accuracy")
worksheet.write('C1', "Precision Micro")
worksheet.write('D1', "Precision Macro")
worksheet.write('E1', "Precision Weighted")
worksheet.write('F1', "F1 Micro")
worksheet.write('G1', "F1 Macro")
worksheet.write('H1', "F1 Weighted")
worksheet.write('I1', "FBeta Micro")
worksheet.write('J1', "FBeta Macro")
worksheet.write('K1', "FBeta Weighted")
worksheet.write('L1', "Recall Micro")
worksheet.write('M1', "Recall Macro")
worksheet.write('N1', "Recall Weighted")
worksheet.write('O1', "Jaccard Micro")
worksheet.write('P1', "Jaccard Macro")
worksheet.write('Q1', "Jaccard Weighted")
worksheet.write('R1', "Matthews CC")
worksheet.write('S1', "Hamming Loss")
worksheet.write('T1', "Time (s)") 

row_spacer = 2


# GO FOR EXECUTIONS    
for algo_name in algo_names:
  current_algo = algo_name
  results = []
  print("")
  print(">>> " + colored(algo_name.upper(), 'yellow') + " <<<")
  for r in range(macro_iterations):
  
    print("--------------------")
    print(algo_name + " - EXECUTION " + colored(str(r+1), 'yellow'))
    
    # This try:/except: bloc has been added because on certain executions the data splitting process results in categories that appear in the test set but not in the training set. In these cases, an error occurs. As we don't want to manipulate the data splitting process, we just cancel this specific execution and go to the next one. The "except:" part (following the "try:" part) just show a message informing of that.
    try:
        # Get time in seconds.
        time_start = time.perf_counter()
        
        # Execute algorithm.
        if algo_name == "csp":
            result = csp()
        elif algo_name == "naive_bayes":
            result = naive_bayes()
        elif algo_name == "logistic_regression":
            result = logistic_regression()
        elif algo_name == "linear_regression":
            result = linear_regression()
        elif algo_name == "neural_network":
            result = neural_network()
        elif algo_name == "neural_net_categorical":
            result = neural_net_categorical()
        elif algo_name == "k_nearest_neighbours":
            result = k_nearest_neighbours()
        elif algo_name == "instance_based":
            result = instance_based()
        elif algo_name == "k_means":
            result = k_means()
        elif algo_name == "support_vector_machine":
            result = support_vector_machine()
        elif algo_name == "decision_tree":
            result = decision_tree()
        elif algo_name == "reinforcement":
            result = reinforcement()

        # Get time in seconds
        time_end = time.perf_counter()
        # Save the execution time of one macro loop.
        exec_time = time_end - time_start
        result.append(exec_time)
        # Add to results
        results.append(result)
    except:
        print("WARNING => EXECUTION " + str(r+1) + " cancelled ! This is probably due to a data splitting issue. Just try to execute again.")
    
    print("--------------------")


  # Get each element separately from results.
  accuracy_values = []
  precision_micro_values = []
  precision_macro_values = []
  precision_weighted_values = []
  f1_micro_values = []
  f1_macro_values = []
  f1_weighted_values = []
  fbeta_micro_values = []
  fbeta_macro_values = []
  fbeta_weighted_values = []
  recall_micro_values = []
  recall_macro_values = []
  recall_weighted_values = []
  jaccard_micro_values = []
  jaccard_macro_values = []
  jaccard_weighted_values = []
  matthews_cc_values = []
  ham_loss_values = []
  time_values = []
  
  for r in results: 
    accuracy_values.append(r[0]["accuracy"])
    precision_micro_values.append(r[0]["precision_micro"])
    precision_macro_values.append(r[0]["precision_macro"])
    precision_weighted_values.append(r[0]["precision_weighted"])
    f1_micro_values.append(r[0]["f1_micro"])
    f1_macro_values.append(r[0]["f1_macro"])
    f1_weighted_values.append(r[0]["f1_weighted"])
    fbeta_micro_values.append(r[0]["fbeta_micro"])
    fbeta_macro_values.append(r[0]["fbeta_macro"])
    fbeta_weighted_values.append(r[0]["fbeta_weighted"])
    recall_micro_values.append(r[0]["recall_micro"])
    recall_macro_values.append(r[0]["recall_macro"])
    recall_weighted_values.append(r[0]["recall_weighted"])
    jaccard_micro_values.append(r[0]["jaccard_micro"])
    jaccard_macro_values.append(r[0]["jaccard_macro"])
    jaccard_weighted_values.append(r[0]["jaccard_weighted"])
    matthews_cc_values.append(r[0]["matthews_cc"])
    ham_loss_values.append(r[0]["ham_loss"])
    time_values.append(r[1])
     
  # Mean Values
  mean_a = round(mean(accuracy_values), 2)
  mean_pmi = round(mean(precision_micro_values), 2)
  mean_pma = round(mean(precision_macro_values), 2)
  mean_pw = round(mean(precision_weighted_values), 2)
  mean_f1mi = round(mean(f1_micro_values), 2)
  mean_f1ma = round(mean(f1_macro_values), 2)
  mean_f1w = round(mean(f1_weighted_values), 2)
  mean_fbmi = round(mean(fbeta_micro_values), 2)
  mean_fbma = round(mean(fbeta_macro_values), 2)
  mean_fbw = round(mean(fbeta_weighted_values), 2)
  mean_rmi = round(mean(recall_micro_values), 2)
  mean_rma = round(mean(recall_macro_values), 2)
  mean_rw = round(mean(recall_weighted_values), 2)
  mean_jmi = round(mean(jaccard_micro_values), 2)
  mean_jma = round(mean(jaccard_macro_values), 2)
  mean_jw = round(mean(jaccard_weighted_values), 2)
  mean_mcc = round(mean(matthews_cc_values), 2)
  mean_hl = round(mean(ham_loss_values), 2)
  mean_t = round(mean(time_values), 2)

  # Save results in excel file.
  worksheet.write('A'+str(row_spacer), algo_name) 
  worksheet.write('B'+str(row_spacer), mean_a)
  worksheet.write('C'+str(row_spacer), mean_pmi)
  worksheet.write('D'+str(row_spacer), mean_pma)
  worksheet.write('E'+str(row_spacer), mean_pw)
  worksheet.write('F'+str(row_spacer), mean_f1mi)
  worksheet.write('G'+str(row_spacer), mean_f1ma)
  worksheet.write('H'+str(row_spacer), mean_f1w)
  worksheet.write('I'+str(row_spacer), mean_fbmi)
  worksheet.write('J'+str(row_spacer), mean_fbma)
  worksheet.write('K'+str(row_spacer), mean_fbw)
  worksheet.write('L'+str(row_spacer), mean_rmi)
  worksheet.write('M'+str(row_spacer), mean_rma)
  worksheet.write('N'+str(row_spacer), mean_rw)
  worksheet.write('O'+str(row_spacer), mean_jmi)
  worksheet.write('P'+str(row_spacer), mean_jma)
  worksheet.write('Q'+str(row_spacer), mean_jw)
  worksheet.write('R'+str(row_spacer), mean_mcc)
  worksheet.write('S'+str(row_spacer), mean_hl)
  worksheet.write('T'+str(row_spacer), mean_t) 
 
  row_spacer += 1
  

# Close the Excel file.
workbook.close()

















