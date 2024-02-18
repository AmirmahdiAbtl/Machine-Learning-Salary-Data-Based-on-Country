# Predicting Salary with Regression Algorithm

+ Data Clearning
+ Feature Engineering
+ Data Visualization
+ Model Training
+ Model Evaluation
+ Implementing Regression from Zero (Gradient Descent, L2 Regularization)
+ Dimantion Reduction
+ Over-Fiting handling....

## Description
This database is about Income and the salary of each job title with features like Country, Region, Age, Years of Experience and Education Level, In this project I tried my best to clean the data such as eliminating the nan values and removing duplicatied values, in adition there are so many wrong value that should be handled and some extra information in some of its columns that I indicated completely in the notebook markdown and i wrote all the story above each code block.

Then there are so many outliers exist in the data that we couldn't remove them because we had only 6k data and there are so valuable for us, for this reason i decided to normalize them with two different way at first i used Logarithmic Normalization and then when I realized that its not able to find this problem completly ( because after normalizing the age column had 12 outliers ) I decided to use Winsorization that i mentioned both the code and theory of this method in project. 

At the next step I found that there is a problem in multi-linear regression algorithm because I took a good result in training accuracy but the test one was really terible, so I decided to use different ways like three regularizaiton way (Lasso, Ridge, Elastic Net) or using the PCA for dimention reduction and .... that i have mentioned them completely in the notebook.

At the End I implemened the Linear Regression :
```python
def compute_cost(x, y, w, b):
    total_cost = 0
    m = len(x)
    for i in range(m):
        f_wb = w * x[i] + b
        total_cost += (f_wb - y[i]) **2
    total_cost = total_cost / (2*m)
    return total_cost
```
```python
def compute_gradient(x, y, w, b):
    m = len(x)
    dj_dw = dj_db = 0
    temp_dw = temp_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        temp_dw += (f_wb - y[i]) * x[i]
        temp_db += (f_wb - y[i])
    dj_dw = temp_dw / m
    dj_db = temp_db / m
    return dj_dw, dj_db
```

```python
def Gradient_descent(x, y, w, b, alpha, epochs):  
    J_history = []
#     p_history = []
    for i in range(epochs):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        J_history.append(compute_cost(x, y, w , b))
#             p_history.append([w,b])
        if i% math.ceil(epochs/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history
```
<h4>At first download <a href="https://drive.google.com/drive/folders/1nWPOFxK3YEh8MVKdWs8A37lXmhbx-7al?usp=sharing">Dataset</a> that we need for doing this project</h4>

