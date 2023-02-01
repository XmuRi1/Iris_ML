# **Iris classifiation**  
My first machine learning project.

**Project tasks:**  
-------------
- Loading a dataset
- Summarizing the dataset
- Data set visualization
- Predictions

***

### **Loading dataset:**  
We use "pandas" to upload our dataset.
```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
```

#### Summarizing the dataset:  

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
