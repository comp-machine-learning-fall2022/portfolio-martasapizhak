# Homework 6

Welcome to your sixth homework assignment for CSC 294. In this assignment, we will 
practice implementing SVM using three kernels. 

In this assignment, we will use the dataset stored in `hw6_data`. This dataset has 
three columns: an `x`-coordinate, a `y`-coordinate, and a label (either 1 or 0). 
Our goal is to determine which kernel for SVM leads to the best classification of 
the data into the two classes.  

A few notes before we begin:
1. For this assignment, please submit your work in an jupyter notebook called 
   `hw6_notebook.ipynb`. 
2. There are unit tests for this assignment, so that you can check that elements of 
   your code work. Place these elements in a file called `hw6.py`. Note that part of 
   the assignment is showing that the tests pass both locally and on travis. 
3. For simplicity, do not make a test directory and just keep your test file in the 
   main `homework-6` directory. 

## Question 1: Projecting data to higher spaces

In this question, we will visualize the impact of two different projections 
on our data. Our investigations will be conducted on both the mean-centered 
and non-mean-centered data.   

### Part A - Centering our data

For this part, create `mean_centered()` data that takes in the data and 
returns data that is centered around the origin (ie. the 0 vector). 

_Note:_ The data for this homework assignment has three columns: 1) an $x$-
value, 2) a $y$-value, and 3) a class label. Your function should take in 
the data with the labels, but the centering should **only** be applied to 
the $x$- and $y$-values.

### Part B - Visualizing projections under our kernels

Recall that the idea behind a kernel is to replace the process of projecting 
our data to a higher space and then computing the SVM. In this part, we will 
construct two different projections that move 2D to 3D:
* Projection 1: $(x_1,x_2)$ --> $(x_1, x_2, \exp{-(x_1^2 +x_2^2)})$  
* Projection 2: $(x_1,x_2)$ --> $(x_1^2,\sqrt{2}x_1x_2, x_2^2)$  

_Hint:_ for projection 1, you may find `np.exp` to be helpful here. 

For this question, create four 3-dimensional plots, one for each of the following:
* Projection 1 on data **without** mean-centering  
* Projection 1 on **mean-centered** data  
* Projection 2 on data **without** mean-centering  
* Projection 2 on **mean-centered** data   

All of your plots should apply color your data based on the class that it 
belongs to. 


_Note 1:_ Projection 1 is the mental picture that many use to understand the 
_radial basis kernel,_ see [this example](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html#Beyond-linear-boundaries:-Kernel-SVM). However, we note that 
projection underlying the RBF is a projection to infinite space (see 
discussion [here](https://stats.stackexchange.com/questions/63881/use-gaussian-rbf-kernel-for-mapping-of-2d-data-to-3d/63909) and 
[here](https://stats.stackexchange.com/questions/109300/applying-an-rbf-kernel-first-and-then-train-using-a-linear-classifier) for more information). 

_Note 2:_ Projection 2 is similar to the projection underlying the 
_polynomial kernel of degree 2._ To learn more about this kernel, see 
[this wikipedia page](https://en.wikipedia.org/wiki/Polynomial_kernel). 

### Part C - Choose a "best" projection

In kernel SVM, our goal is to project our data into a higher dimensional 
space such that the classes are well separated. Examine your four plots 
and decide which one offers the best resulting projection for splitting 
the data with a rigid plane (as in a wall dividing the 3D space). In your 
answer, you must state: 1) which projection (1 or 2) you are selecting 
and 2) whether the data needs to be mean-centered or not. Justify your 
choice. 


## Question 2: Perform 10-fold Cross-Validation

Create a function `ten_fold` that performs a 10-fold cross validation for 1) a 
given dataset, 2) a specified kernel, and 3) a flag that tells you whether 
to mean-center your data. Your function should only output the cross-validation 
error and nothing else. 

Perform 10-fold cross-validation to determine which kernel is the appropriate one 
for our SVM: 
* Linear kernel (ie. the usual dot-product one)
* Radial basis kernel (ie. the one with a projection similar to projection 1)
* Polynomial kernel of degree 2  (ie. the one with a projection similar to projection 2)

For the later two, either pre-process your data with mean-centering or not, 
based on your answer from question 1C. 

Select one of your kernels as the appropriate one. Justify your choice. 

## Question 3: Tune your model 

Once you have selected your kernel, use all of the data to tune the SVM using 
your selected kernel. Show the results in a 2-D plot that shows both the 
decision boundary and the margins. Note your observations about this plot in 
two to four sentences. 

## Question 4: Explore new data

There is some new data contained in the `test.csv`. Using your kernel 
SVM determine which classes your classifier thinks they should belong to. 
Compare them to where they are labeled to belong.

Show this validation data on the same plot as the decision boundary and the 
margins _without_ the data used to tune the model. 

## Question 5: Tests passing 
For this question, please submit screenshots of your tests passing locally and 
on travis. 

## Rubric

* Question 1: 14 points total
   * 2 pts for centering data function
   * 8 pts for plots
   * 4 points for projection selection and justification
* Question 2: 18 points total
   * 8 pts for `ten_fold` definition
   * 6 pts for applying to the three kernels
   * 4 points for kernel selection and justification
* Question 3: 8 points total
   * 4 pts for tuning
   * 4 pts for plot and observations
* Question 4: 8 points total 
   * 4 pts for predictions on validation set
   * 4 pts for plot and observations
* Question 5: 4 points total 

## Rubric

|  Q  | Topic                         | No Attempt | Partial | Complete | 
|-----|-----------------------------  |------------|---------|----------|
|  1  | Mean Center the data          |            |         | 1        | 
| ... | Plot projected the data       |            |         | 1        |
| ... | Select and justify projection |            |         | 1        |
|  2  | Implement ten_fold            |            |         | 1        |   
| ... | Apply to three kernels        |            |         | 1        |
| ... | Select and justify kernel     |            |         | 1        |
|  3  | Tune SVM                      |            |         | 1        |
| ... | Plot tuned SVM                |            |         | 1        | 
| ... | Discuss the resulting plot    |            |         | 1        |
|  4  | Test final model on test set  |            |         | 1        |
| ... | Plot and discuss your results |            |         | 1        |
|  5  | Local tests                   |            |         | 1        |
| ... | GitHub Actions                |            |         | 1        |

|  Q  | Topic                         | Have questions about| Could again without help | 
|-----|-----------------------------  |---------------------|--------------------------|
|  1  | Mean Center the data          |                     | +                        |
| ... | Plot projected the data       |                     | +                        |
| ... | Select and justify projection |                     | +                        |
|  2  | Implement ten_fold            |                     | +                        |
| ... | Apply to three kernels        |                     | +                        |
| ... | Select and justify kernel     |                     | +                        |
|  3  | Tune SVM                      |                     | +                        |
| ... | Plot tuned SVM                |                     | +                        |
| ... | Discuss the resulting plot    |                     | +                        |
|  4  | Test final model on test set  |                     | +                        |
| ... | Plot and discuss your results |                     | +                        |
|  5  | Local tests                   |                     | +                        |
| ... | GitHub Actions                |                     | +                        |

### Reminders
* Check your file names carefully. Be sure to set up your directory to ignore and connect 
when needed
* Any import statements should be at the top of your python files or in the first 
code block of a notebook. 

#### Resources consulted for this homework
0. [In-Depth: Support Vector Machines](https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html)
1. [Polynomial kernel on Wikipedia](https://en.wikipedia.org/wiki/Polynomial_kernel)
2. [Support Vector Machine (SVM) Tutorial](https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93)   
3. [Applying an RBF kernel first and then train using a Linear Classifier](https://stats.stackexchange.com/questions/109300/applying-an-rbf-kernel-first-and-then-train-using-a-linear-classifier)  
4. [Use Gaussian RBF kernel for mapping of 2D data to 3D](https://stats.stackexchange.com/questions/63881/use-gaussian-rbf-kernel-for-mapping-of-2d-data-to-3d/63909)
5. [`numpy.isclose` helpfile](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)  
6. [`numpy.unique` helpfile](https://numpy.org/doc/stable/reference/generated/numpy.unique.html)
