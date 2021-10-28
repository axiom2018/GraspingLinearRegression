import numpy as np
import matplotlib.pyplot as plt

''' This is called LinearRegressionComplex because it's a bit more difficult to grasp in depth but it's doable.
    Also this definitely relates more to the version Ayush taught us in the lecture, so hopefully this will clear
    some of that up for anyone who had trouble.

    1) m_weights & m_bias - Remember that the main equation for Linear Regression is y=mx+b, which can also be 
        written as y=wx+b. W = weights, and b = bias in this equation.

'''
class LinearRegressionComplex:
    def __init__(self):
        self.m_weights = None
        self.m_bias = None


    def TrainModel(self, X, y, epochs, learningRate):
        ''' Get the rows & columns (features) immediately, they're both needed. The .shape will be (100, 1)
            when this code is ran as is because there are 100 rows and 1 column (feature). '''
        rows, features = X.shape

        ''' Before the algorithm starts, we have to set default values to the weights and bias. It's clear that
            every feature has an associated weight so use np.zeros to get literally a weight for each feature.
            That's why that value was saved above. Ex: If we have any dataset and it has 4 features, then put
            np.zeros(4) to get 4 zeros to start.
            
            Bias as well starts as 0. '''
        self.m_weights = np.zeros(features)
        self.m_bias = 0

        ''' The meat of the algorithm. Normally loops occur with a variable such as "for i in range" but
            that's omitted because it wasn't needed.
        
            1) Predict the y value.
                Using the dot function, multiply the X values (default was 100 of them) by the amount
                of m_weights (default was 1 weight). Take EVERY value in X and multiply it with EVERY
                value in m_weights I believe. Seems true because the yHat value will be 100 values in
                length afterwards. Matching the length of X.


            2) Calculate the Derivatives with respect to w and bias.

                -----For w-----

                The transpose of X is used. What does it do? For example if X's shape is (100, 1),
                that means it has 100 rows and 1 column. But transpose flips that around so the 
                new shape is (1, 100). So with the transpose variable we can index it like array[0] 
                and still get all 100 values in that array at index 0. To access the values 1 by 1 
                in a transpose array just do array[0][0], array[0][1], etc.

                So when the dot function is used when the derivates are figured out (See line with 
                variable "dw"). The tranpose of X is used and despite X having 100 rows, for giving
                a visual example I'll use 5. It starts out as:

                        0
                0       32.50
                1       53.42
                2       61.53
                3       47.47
                4       59.81

                Then it becomes
                        0       1       2       3       4 
                0   32.50   53.42   61.53   47.47   59.81

                So we get the dot product, of this new transposed vector above that is multipled against
                the difference of the predicted value (yHat) and the actual value (y). Both y and yHat 
                have the same shape.
                
                
                Then of course to update the weights and bias, follow the equations:
                    a) w = w - learningRate * derivative of w
                    b) b = b - learningRate * derivative of b

        '''
        for _ in range(epochs):
            yHat = np.dot(X, self.m_weights) + self.m_bias

            dw = (1/rows) * np.dot(X.T, (yHat - y))
            db = (1/rows) * np.sum(yHat - y)

            self.m_weights = self.m_weights - learningRate * dw
            self.m_bias = self.m_bias - learningRate * db


    
    ''' Remember to get a predicted value in LinearRegression, y=mx+b is used (in this case it's
        y=wx+b) So all that's passed in is an X '''
    def Predict(self, X):
        return np.dot(X, self.m_weights) + self.m_bias
    

    def ShowWeightsAndBias(self):
        print(f'Weights:\n{self.m_weights}.\nBias:{self.m_bias}.')


    def ScatterPlot(self, points, predictedValues):
        plt.scatter(points.iloc[:, 0], points.iloc[:, 1], color='black')
        plt.plot(points.studytime, predictedValues, color='red')
        plt.show()
