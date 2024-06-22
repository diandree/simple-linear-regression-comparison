#--------------------------- IMPORTS ----------------------------------------------
import os

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#---------------------------------------------------------------------------------------
#-------------------------------- DATASET ----------------------------------------------

#Reading dataset into a DataFrame data structure
df = pd.read_csv("Salary_Data.csv")
print(f'Dataset:\n{df}\n')

#Verifying if there are null values on the dataset
print(f'Number of null data:\n{df.isnull().sum()}')

#Data description
print(f'\n{df.describe()}')

# Converting our DataFrame structure into NumPy arrays
X_set = df['YearsExperience'].to_numpy()
y_set = df['Salary'].to_numpy()

print(f'X_set shape: {X_set.shape}\ny_set shape: {y_set.shape}')
print(f'\nX_set details:\n\tCount: {X_set.shape[0]}\n\tMax value: {np.max(X_set)}\n\tMin value: {np.min(X_set)}\n\tStd: {np.std(X_set)}\n\tMean: {np.mean(X_set)}')

#Data distribution - X_set
plt.hist(X_set, color="blue")
plt.xlabel('Years of experience')
plt.ylabel('Count')
plt.title(f'Years of Experience Distribution')
plt.show()

print(f'Y_set details:\n\tCount: {y_set.shape[0]}\n\tMax value: {np.max(y_set)}\n\tMin value: {np.min(y_set)}\n\tStd: {np.std(y_set)}\n\tMean: {np.mean(y_set)}')

#Data distribution - y_set
plt.hist(y_set, color="red")
plt.xlabel('Salary')
plt.ylabel('Count')
plt.title(f'Salary Distribution')
plt.show()

#Relation between years of experience and salary
plt.plot(X_set, y_set, 'mo')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.show()

#-------------------------------- SIMPLE LR IMPLEMENTATION ----------------------------------------------

#compute_cost implementation
def compute_cost(X, y, w, b):
    cost = 0.
    m = X.shape[0]

    cost = np.sum((w * X + b - y) ** 2) / (2 * m)

    return cost

#compute_gradient implementation
def compute_gradient(X, y, w, b):
    m = X.shape[0] #Get the number of rows
    
    #Variables initialization
    dj_dw = 0.
    dj_db = 0.

    for i in range(m):
        err = (w*X[i] + b) - y[i]
        dj_dw = dj_dw + (err * X[i])
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

#gradient_descent implementation
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_hist = [] 
    w = w_in  
    b = b_in

    for i in range(num_iters):

        dj_db,dj_dw = compute_gradient(X, y, w, b) #Calculate the gradient for w and b

        # Parameters update using w, b, alpha and gradient
        w -= (alpha * dj_dw)               
        b -= (alpha * dj_db)             

        # Save cost J at each iteration during training
        '''
            Note: In this case, we will not interact with the cost values obtained during training,
            but it is important to track these cost values to evaluate the model's performance.
        '''
        if i < num_iters: #Prevent resource exhaustion 
            J_hist.append(compute_cost(X, y, w, b)) 

    return w, b 

#R2 score implementation
def score(y, preds):
    v = np.sum((y - np.mean(y)) ** 2) # Calculate the total sum of squares
    u = np.sum((y - preds) ** 2) # Calculate the residual sum of squares (RSS)
    r2 = 1 - (u / v) # Calculate the R^2 score

    return r2

#K fold cross validation implementation
def k_fold_cross_validation(X_set, y_set, k, alpha, num_iters):
    np.random.seed(42)  # Set seed for reproducibility
    
    #Converting input features and target variable into 1-dimensional arrays
    X = X_set.flatten()
    y = y_set.flatten()
    
    indices = np.arange(X.shape[0]) #Create an array of indices corresponding to the number of data points
    np.random.shuffle(indices) #Shuffle the indices randomly to ensure randomness in data partitioning.
    fold_size = len(X) // k #Calculate the size of each fold
    
    J_hist = []
    r2_scores = []
    
    #Stores the best values found during training, so they can be used on the prediction 
    final_info = {
        'max_r2_score': 0.,
        'b_final': 0.,
        'w_final': 0.,
        'k': 0
    }
    
    
    for i in range(k):
        test_indices = indices[i * fold_size:(i+1) * fold_size] #Select indices for the current test set
        train_indices = np.setdiff1d(indices, test_indices) #Select indices for the training set by excluding the test set indices

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # initialize parameters
        initial_w = np.random.randn()
        initial_b = np.random.randn()

        # run gradient descent 
        w, b = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, num_iters)
        print(f'b,w found by gradient descent:\n\tb = {b:0.2f}\n\tw = {w}\n')

        preds = (X_test * w) + b
        
        #Presenting predictions vs actual targets
        df = pd.DataFrame({'Actual': y_test, 'Predict': preds})
        print(df)

        #Obtaining R^2 score
        r2 = score(y_test, preds)
        print(f'\n[k={i}] R^2 score: {r2:.4f}')
        r2_scores.append(r2)

        #Store the parameters that presented the best R2 performance
        if r2 > final_info['max_r2_score']:
            final_info['max_r2_score'] = r2 
            final_info['w_final'] = w
            final_info['b_final'] = b
            final_info['k'] = i
        
        J_hist.append(compute_cost(X_test, y_test, w, b))
        print(f'[k={i}] Cost: {J_hist[-1]:.2f}')
        

        #Plotting model fitted line 
        plt.plot(X_test, y_test, 'bo')
        plt.plot(X_test, preds, 'r')
        plt.xlabel('Years of experience')
        plt.ylabel('Salary')
        plt.title(f'Fitted Line Plot k={i}')
        plt.legend(['Actual Data', 'Fitted Line'])
        plt.show()
        
    return final_info, np.mean(r2_scores), np.std(r2_scores), J_hist


#Run k fold cross validation
final_info, mean_r2, std_r2, J_hist = k_fold_cross_validation(X_set, y_set, 5, 0.007, 100000)

print(f"Mean R^2: {mean_r2:.5f}")
print(f"Standard Deviation of R^2: {std_r2:.5f}")

def prediction(input, b, w):
    return (w * input) + b

            
print(f'At k={final_info["k"]} with R2 score={final_info["max_r2_score"]:0.4f}, b,w found by gradient descent:\n\tb = {final_info["b_final"]:0.2f}\n\tw = {final_info["w_final"]:0.2f}')
print(f'For 10.3 years of experience, this is the estimated salary: {prediction(10.3, final_info["b_final"], final_info["w_final"]):0.2f}') 

#-------------------------------- SKLEARN IMPLEMENTATION ----------------------------------------------

df = pd.read_csv("Salary_Data.csv")

#Splitting variables
X = df.iloc[:, :1]
y = df.iloc[:, 1:]

#Splitting dataset into test-train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'X train: {X_train.shape}\ny train: {y_train.shape}\nX test: {X_test.shape}\ny test: {y_test.shape}')

#Defining the model
regressor = LinearRegression().fit(X_train, y_train)
print(f'\nR^2 score (training set): {regressor.score(X_train, y_train)}')


#Making predictions
y_pred_test = regressor.predict(X_test)
sk_r2_score = regressor.score(X_test, y_test)
print(f'R^2 score (testing set): {sk_r2_score}')

mse = mean_squared_error(y_test, y_pred_test)
print(f'Mean Squared Error: {mse}')

# Regressor coefficients and intercept
coef = regressor.coef_
intercept = regressor.intercept_

print(f'\nCoefficient (w): {coef[0][0]:.2f}')
print(f'Intercept (b): {intercept[0]:.2f}')


# Prediction on testing set
plt.scatter(X_test, y_test, color ='blue', label='Actual Data')
plt.plot(X_test, y_pred_test, color = 'red', label='Fitted Data')
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['Actual Data', 'Fitted Line'])
plt.show()

#-------------------------------- COMPARISON ----------------------------------------------

print(f'\nw:\tSimple LR={final_info["w_final"]:0.2f}\t\t\tSklearn={coef[0][0]:.2f}')
print(f'b:\tSimple LR={final_info["b_final"]:0.2f}\t\t\tSklearn={intercept[0]:.2f}')
print(f'MSE:\tSimple LR={J_hist[final_info["k"]]:0.2f}\t\t\tSklearn={mse:.2f}')
print(f'R2:\tSimple LR={final_info["max_r2_score"]:0.3f}\t\t\t\tSklearn={sk_r2_score:.3f}')