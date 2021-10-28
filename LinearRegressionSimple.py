import matplotlib.pyplot as plt

''' This is named LinearRegressionSimple because it's the simplest way I've found how to do it.

    1) m_totalError - Used in the manually written cost function called "LossFunction". Loss function, cost function, both
        mean the same thing. Mean square error is used

    2) m_bestFitLineXValues & m_bestFitLineYValues - These are lists that will contain the points to CREATE a best fit line.
        Remember that the equation y=mx+b is used in ORDER to get/predict a y value, hence the 'y' at the START of y=mx+b.
        The m & c values are needed to complete this equation and even attempt to FIND the best fit line. Therefore it's clear
        that these have to be found. But when they are found, just plug them in to the equation y=mx+b. These new y values,
        found by that equation, will go into the m_bestFitLineYValues list and later be used to plot. '''
class LinearRegressionSimple:
    def __init__(self):
        self.m_totalError = 0
        self.m_bestFitLineXValues = []
        self.m_bestFitLineYValues = []

    ''' The Loss function measures how far the predicted values are from the actual values. Very important
        in any model. 
        
        The error is calculated using mean square error (mse), very simple. 4 simple steps to do this.
        1) Get the difference between the ACTUAL value, and the PREDICTED value.
        2) Square that difference.
        3) Get the squares of the remaining actual values and predicted values.
        4) Get the mean of all the values. 
        

        ---iloc, and what it does---

        a) iloc - This is used in the dataframe to get access to a certain row. Example:

            studytime     scores
            0   32.502345  31.707006
            1   53.426804  68.777596
            2   61.530358  62.562382

            Let's say we have df.iloc[0], then that will access the first row at index 0 which has "32.502345  31.707006".
            BUT how do we access one of those specific values? Just type in the column names like df.iloc[0].studytime will
            get "32.502345". So how do we access the bottom right value "62.562382"? Like df.iloc[2].scores. That simple.
            We'll use this to get every data point basically on both the x and y axis.  


        b) iloc[:, 0] - Use the studytime/scores dataset above as an example. Using this [:, 0] says "Get me ALL the values
            in column/feature 0". That simple. '''
    def LossFunction(self):
        for i in range(len(self.m_dataframe)):
            x = self.m_dataframe.iloc[i].studytime
            y = self.m_dataframe.iloc[i].scores

            ''' "(y - (self.m_m * x + self.m_b))" is where we find the difference from the actual value (y) and predicted value (mx+b). 
                Remember the mx+b part in the equation is literally going to predict a y value FOR us. So that's why this works. 
                Also the "**2" is the square root. This has to be squared because the data point can be above or below the line. '''
            self.m_totalError = self.m_totalError + (y - (self.m_m * x + self.m_b)) ** 2

        return self.m_totalError / float(len(self.m_dataframe))


    ''' I tried to do this function without needing to pass so many arguments but ended up getting a horrible m and b value which
        in turn gave a horrible line. 

        dm & db - Partial derivates of m and b. The actual math formula is still a bit tricky to understand with ease but I'm 
            getting better at it the more I go over it. These are updated in the for loop and the primary thing to remember is
            that the actual y and the predicted y (which is mx+b) comes in at the end.

        x & y - The points to be used. In main.py, it's explained that iloc can be used to access a point in the x feature column
            and the y feature column. That's what is happening here. Row by row.

        m & b - After the loop apply the updates to both m and b which is:
            m = m - learning rate * partial derivative of m
            b = b - learning rate * partial derivative of b.
    
    '''
    def GradientDescent(self, curM, curB, points, learningRate):
        dm = 0
        db = 0
        n = len(points)

        for i in range(n):
            x = points.iloc[i].studytime
            y = points.iloc[i].scores

            dm += -(2/n) * x * (y - (curM * x + curB))
            db += -(2/n) * (y - (curM* x + curB))

        m = curM - learningRate * dm
        b = curB - learningRate * db
        
        return m, b



    ''' Remember to be able to predict, we'll use the equation y=mx+b. So giving this function an x value
        AFTER the m & b values were found with gradient descent, will predict a new y value. '''
    def Predict(self, m, x, b):
        yHat = (m * x) + b
        return yHat
    


    # This function will try to get the proper points to create the best fit line.
    def CreateBestFitLine(self, points, m, b):
        # The x values will be the same as the x axis so that takes care of itself.
        self.m_bestFitLineXValues = points.iloc[:, 0]
        ''' List comprehension is used between [], it's a 1 line for loop. For every number in the studytime feature/column,  
            call the function Predict and pass it that value (along with the other 2 variables). '''
        self.m_bestFitLineYValues = [self.Predict(m, val, b) for val in points.studytime]



    # Display scatter plot.
    def ScatterPlot(self, points):
        plt.scatter(points.iloc[:, 0], points.iloc[:, 1], color='green')
        plt.plot(self.m_bestFitLineXValues, self.m_bestFitLineYValues, color='red')
        plt.show()