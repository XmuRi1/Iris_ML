# **Iris classifiation**
My first machine learning project.

**Project tasks:**
-------------
- Loading a dataset
- Summarizing the dataset
- Data set visualization
- Predictions

***

#### **Loading dataset:**
We use "pandas" to upload our dataset.
```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
```

#### **Summarizing the dataset:**

In this step, we'll look at the data in several different ways:
1. Dataset sizes
2. Look at the data itself
3. A statistical summary of all attributes
4. Breakdown data by class variable

**Dataset sizes**  
We can quickly understand how many instances (rows) and how many attributes (columns) the data contains using the shape property.
```python
shape
print(dataset.shape)
```
```python
(150, 5)
```

**Look at the data itself**
```python
head
print(dataset.head(20))
```
```python
    sepal-length  sepal-width  petal-length  petal-width        class
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
10           5.4          3.7           1.5          0.2  Iris-setosa
11           4.8          3.4           1.6          0.2  Iris-setosa
12           4.8          3.0           1.4          0.1  Iris-setosa
13           4.3          3.0           1.1          0.1  Iris-setosa
14           5.8          4.0           1.2          0.2  Iris-setosa
15           5.7          4.4           1.5          0.4  Iris-setosa
16           5.4          3.9           1.3          0.4  Iris-setosa
17           5.1          3.5           1.4          0.3  Iris-setosa
18           5.7          3.8           1.7          0.3  Iris-setosa
19           5.1          3.8           1.5          0.3  Iris-setosa
```

**A statistical summary of all attributes**  
It shows the count, average, minimum and maximum values, as well as some percentiles.

```python
descriptions
print(dataset.describe())
```
```python
       sepal-length  sepal-width  petal-length  petal-width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```

**Breakdown data by class variable**  
We can look now at the number of instances (rows) that each class belongs to. We can consider this as an absolute score.

```python
class distribution
print(dataset.groupby('class').size())
```
```python
class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
```


#### **Data visualization**  

We will consider two types of plots:  
1. One-dimensional plots to better understand each attribute.  
1. Multidimensional plots for a better understanding of relationships between attributes.  

**One-dimensional plots**

We'll start with a few one-dimensional plots that will give us a clearer idea of the distribution of the input features:

```python
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
```

![](https://github.com/XmuRi1/Iris_ML/blob/main/images/Figure_1.png)

We can also create a histogram of each input variable to get an idea of the distribution.

```python
dataset.hist()
plt.show()
```

![](https://github.com/XmuRi1/Iris_ML/blob/main/images/Figure_2.png)

**Multidimensional plots**

Now we can look at the interaction between variables.
First, let's look at the scatterplots of all attribute pairs. This can be useful for defining structured relationships between input variables.

```python
scatter_matrix(dataset)
plt.show()
```

![](https://github.com/XmuRi1/Iris_ML/blob/main/images/Figure_3.png)

#### **Algorithm evaluation**

First of all, we will split the uploaded dataset into two, 80% of which we will use to train our models and 20% we will keep as a validation dataset.
We will also define data for training and validation.

```python
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
```

We will use 10-fold cross-validation to evaluate accuracy.
'accuracy' is a metric for evaluating an algorithm.

```python
seed = 7
scoring = 'accuracy'
```

We will evaluate 6 different algorithms:
- Logistic Regression (LR)
- Linear Discriminant Analysis (LDA)
- K-Nearest neighbors (KNN)
- Classification and Regression Trees (CART)
- Gaussian Naive Bayes (NB)
- Support Vector Machines (SVM)

It is a good mix of simple linear (LR and LDA), non-linear (KNN, CART, NB and SVM) algorithms. We reset the seed of random numbers before each run to ensure that each algorithm is evaluated using exactly the same data and to make sure the results are directly comparable.

```python
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# each models evaluation
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
```

We now have 6 models and accuracy scores for each. We need to compare the models with each other and choose the most accurate ones.

```python
LR: 0.983333 (0.033333)
LDA: 0.975000 (0.038188)
KNN: 0.983333 (0.033333)
CART: 0.975000 (0.038188)
NB: 0.975000 (0.053359)
SVM: 0.983333 (0.033333)
```

It looks like LR, KNN and SVM have the same high accuracy score.
We can also plot the results of the model evaluation and compare the spread and average accuracy of each model.
There are many accuracy metrics for each algorithm because each algorithm has been evaluated 10 times (10 times cross-validation).

```python
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
```

![](https://github.com/XmuRi1/Iris_ML/blob/main/images/Figure_4.png)

You can see many samples are reaching 100% accuracy.

#### **Predictions**

For example, let's take one of the most accurate algorithms - Knn.
We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, confusion matrix, and classification report.

```python
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
```

We can see that the accuracy is 0.9 or 90%. The confusion matrix gives an idea of the three mistakes made. Finally, the classification report provides a breakdown of each class by accuracy, recall, f1-score and support, showing excellent results.

```python
0.9

[[ 7  0  0]
 [ 0 11  1]
 [ 0  2  9]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00         7
Iris-versicolor       0.85      0.92      0.88        12
 Iris-virginica       0.90      0.82      0.86        11

       accuracy                           0.90        30
      macro avg       0.92      0.91      0.91        30
   weighted avg       0.90      0.90      0.90        30
   ```
   



----------------------
