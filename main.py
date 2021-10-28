import pandas as pd
from LinearRegressionSimple import LinearRegressionSimple
from LinearRegressionComplex import LinearRegressionComplex
import numpy as np
import os

''' This project is for demonstrating in as much detail as I can how Linear Regression works. 
    Hopefully the other students benefit from this. There are several classes included in this project
    that work with the same dataset but they all do Linear Regression as that is the focus. the classes
    are heavily commented but of course if anyone has any concerns or questions than I'll explain. 

    - Omar Moodie
'''


df = pd.read_csv('data.csv', names=['studytime', 'scores'])



'''                                     Example 1

    Create the first class object (the class that achieves Linear Regression in a much more simple way
    than the other one) and then set all the necessary variables for linear regression to work.

    1) m & b - Part of the equation of this model, y=mx+b. This can also be seen as y=mx+c, y=wx+b, 
        y=wx+c, and probably more examples but they mean the same thing. m is the slope and b is the
        y-intercept.

    2) epochs - Basically iterations, that's all. How long we'll keep trying the gradient descent 
        optimization.
    
    3) learningRate - This determines how MUCH the variable m changes each iteration. Play with this for
        different outputs!

    The for loop begins in order to keep updating m and b. df is of course the dataframe and the function
    will extract the points to graph. Display m and b too.

    Then create the best fit line after the values m and b are found and show the plot, clear the screen to 
    show the next
'''
lr = LinearRegressionSimple()
m = 0
b = 0
epochs = 300
learningRate = 0.0001

for i in range(epochs):
    m, b = lr.GradientDescent(m, b, df, learningRate)

print(f'm: {m}. b: {b}.')

lr.CreateBestFitLine(df, m, b)
lr.ScatterPlot(df)
os.system('cls')



'''                                     Example 2

    A different way to do it. This one relates to Ayush's assignment.
    

    1) X - The data is split into 2 features/columns, studytime and scores. We'll simply make an array
        with np.array but then reshaping it. Why? Because np.array will give us a regular 1d list. 
        We specifically reshape to 100, because there's 100 rows in the dataset. Then set it to 1 column
        because this is the data that will be used to predict a y value, it's one feature in itself. 
        Hopefully that explains it.


    2) y - This is the target variable, what we'll be trying to predict. Of course this will play a major 
        part in creating the best fit line too.


    3) TrainModel - Literally just the gradient descent algorithm under the hood.


    4) ShowWeightsAndBias - Visually see the values w & bias (aka m and b).


    5) Predict(X) - Remember that the equation in Linear Regression is y=mx+b (in this case it's y=wx+b)
        but the point is simple. Just toss in X values in the function, AFTER it already figured out
        the weights and bias from the TrainModel function, to get all the predicted values..


    6) ScatterPlot - Shows the points and the best fit line in one.

    Important: Converting df.studytime/scores to ndarrays. By default they're pandas.core.series.Series. 
    The other working implementation of the complex Linear Regression only works when X shape is (n, 1) 
    and the y is (n,). Shapes '''
X = np.array(df.studytime).reshape(100, 1)
y = np.array(df.scores)

lrc = LinearRegressionComplex()
lrc.TrainModel(X, y, epochs, learningRate)
lrc.ShowWeightsAndBias()
predictedValues = lrc.Predict(X)
lrc.ScatterPlot(df, predictedValues)