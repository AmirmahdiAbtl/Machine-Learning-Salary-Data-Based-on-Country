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

<h4>At first download <a href="https://drive.google.com/drive/folders/1nWPOFxK3YEh8MVKdWs8A37lXmhbx-7al?usp=sharing">Dataset</a> that we need for doing this project</h4>
