# Problem Statement

## 1. Objective
The objective is to predict or estimate the salary of an individual based on the number of years of professional experience.

## 2. Motivation
This predictive modeling task is relevant in various sectors such as human resources, recruitment, and workforce management, where understanding salary expectations based on experience can inform decisions on compensation, career development, and resource allocation.

## 3. Dataset

![Dataset snapshot](.\assets\about-dataset.PNG "Dataset snapshot")


The dataset consists of historical records where each entry includes:
* YearsExperience: This is a continuous numerical variable representing the number of years of professional experience.
* Salary: This is the target variable, representing the actual salary earned by individuals corresponding to their years of experience.

You can find this dataset on [Kaggle](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)

## 4. Methodology

This problem will be addressed using two different methods:
1. Implementing a Simple Linear Regression model from scratch.
2. Implementing a Simple Linear Regression using the sklearn.linear_model.LinearRegression class.

### 1. Tools

* NumPy: A library for scientific computing, mainly involving linear algebra operations.
* Pandas: A library for data analysis and manipulation.
* Matplotlib: A library for plotting data.
* Scikit-learn (Sklearn): A machine learning library that provides simple and efficient tools for data analysis and machine learning tasks.



## 1. Gradient descent summary
A linear model that predicts $f_{w,b}(x^{(i)})$:
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$
In linear regression, we utilize input training data to fit the parameters $w$,$b$ by minimizing a measure of the error between our predictions $f_{w,b}(x^{(i)})$ and the actual data $y^{(i)}$. The measure is called the $cost$, $J(w,b)$. In training you measure the cost over all of our training samples $x^{(i)},y^{(i)}$
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2\tag{2}$$ 

But for this implementation, we will also analyse the *coefficient of determination*, or $R^2$. There are a number of variants, but the following one is widely used:

$$R^2 = \frac{\text{sum squared regression (SSR)}}{\text{total sum of squares (SST)}} = \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}\tag{3}$$ 

 *gradient descent* is described as:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{4}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$
where, parameters $w$, $b$ are updated simultaneously.  
The gradient is defined as:
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{5}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{6}\\
\end{align}
$$

Here *simultaniously* means that you calculate the partial derivatives for all the parameters before updating any of the parameters.

## 1.2 Gradient Descent Implementation

Implementing gradient descent algorithm for one feature. We will need the following functions: 
- `compute_cost` implements equation (2)
- `compute_gradient` implements equation (5) and (6) above
- `score`: implements equation (2)
- `gradient_descent`, implements equation (4)

*Note: Since we have a limited dataset, we are going to use a resampling procedure called cross-validation. This procedure has a single parameter called k, which refers to the number of groups into which a given data sample is to be split. As such, the procedure is often called k-fold cross-validation.*
