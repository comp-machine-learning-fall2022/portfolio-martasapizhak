## Summary
This element explores supervised learning algorithm SVM and some concepts 
relating to supervised machine learning algorithms. The material presented 
in the directory is accounted for audience with some exposure to ML concepts.

In the first part of the notebook, I introduce classification problem and 
a k-means approach for solving it.

In the second part, I explore SVM as a supervised classification algorithm. 

Overall the aim of this element is to present additional way to classify data, walking 
the reader through the idea of classification and introducing a supervised algorithm for it.

## Files explained
Many of additional comments and explanations for this notebook are provided
in the _funcs_classification.py_.

* data_classification.csv     contains data we will classify in this element
* funcs_classification.py     is a commented file with functions we will use for classification
* supervised.ipynb            contains a walkthrough the material presenting
* test.csv                    extra data to evaluate model
* test_classification.py      unit tests for some functions used in the notebook


## Exploring ethical implications of ML
While classification is a great technique for solving problems with machine learning,
the results it presents us with may have many possible caveats.
Consider the following short summary of aspects that need extra attention when classification
is applied. 
1. While high accuracy helps to complete tasks better, it can pose risks to
 privacy. Because it requires a lot of data to create an accurate model it may be targeted
at people or force invasive access to private personal data. 
2. Predictions made with ML tools are highly dependent on training data and if applied without
additional investigation, it may be highly biased and targeted towards something specific.
3. When the outcome of ML research influences important decisions, it's important to ensure 
that the data it's trained on is reliable and data does not provide information to 
people with wrongful intentions.
4. It's important that when data is studied it is clear how it is collected and what it
is used for. The resources and research should be traceable -- ML explorers should document their goals, definitions, 
design choices, and assumptions. They also should communicate potential gaps in understanding
or use cases. It should be interpretable, explainable in human terms to avoid the black-box problem which may cause
inappropriate usage of research results.
5. It's important that the ML researchers take accountability when their models are used in high 
impact areas (healthcare, welfare, criminal justice) and that harms can be large when they 
go wrong
6. Finally, computationally complex ML approaches trained on very large data sets can 
have a large environmental impact, given the amount of energy required to power 
the training phase.

“Algorithms and the data that drive them are designed and created by people – there is 
always a human ultimately responsible for decisions made or informed by an algorithm. 
"The algorithm did it" is not an acceptable excuse if algorithmic systems make mistakes 
or have undesired consequences, including from machine-learning processes.” [FATML]

It's important to keep these things in mind at each stage of ML research and applications.
It can be done through analyzing potential intentions and implications at each stage,
making sure that data is kept secure and the developer include diverse perspectives in the 
process.

## resources consulted for this file:
https://www.w3.org/TR/webmachinelearning-ethics/





TODO:

add run time in unsup


