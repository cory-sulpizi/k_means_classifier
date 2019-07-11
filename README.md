# K-Means Classifier
A classifier that uses k-means clustering to define the decision boundary for each class.

# train() Function
Calculates the weights (w), softmax bias (b), and k-cluster centres (g) required for predicting classes.

Required Inputs: 
    x -> shape(n, m). n data points with m-dimensional coordinates.
    y -> shape(n, ). n data points with class between 0 and v-1, where v is the 
            number of possible classes.
    k -> shape(v, ). The number of k-cluster centres desired for each class.

Optional Inputs:
    vb -> Verbose. Whether or not the function should print its results
    loss_coef ->  shape(v, ). How much the loss of each class should 
            be scaled by. For example, loss_coef = [1,2] means that the loss 
            of class 0 is half as important as the loss of class 1.
    learn_rate -> Learning rate for stochastic gradient descent.
    max_iters -> The number of iterations for training.
    sgd_size -> The batch size for stochastic gradient descent.

Outputs:
    w_out -> shape(sum(k), v). The predicted weight values.
    b_out -> The bias of the model's softmax function.
    g -> shape(sum(k), m). The found k-cluster centres.
    acc -> The overall training accuracy of the model.
    acc_by_label -> shape(v). The training accuracy for each class.
      
# Finding k-values
There are three methods for finding the k-values for the model. 
1) Visual inspection. Use a scatter plot to plot your data points, and decide approximately how many clusters you think would be appropriate. This is less accurate than method 2, and only really works for 2D points (since visualizing in 3D or higher is much harder).
2) Guess the values. This is much less accurate than the other two methods, but can work fine for quick analysis.
3) Try a variety of combinations for k. Input the first combination, train the network, record the accuracy, then repeat with the second combination. Choose the k-values that return the highest testing accuracy. 

# Loss Coefficients
The default loss coefficient is 1. What this means is that every data point is weighted equally when calculating the loss function. This means that by default the model is attempting to find the highest overall accuracy. One issue with selecting the default coefficient is that it generally leads to high accuracies in classes with lots of data points, but much lower accuracies for classes with fewer data points. 

The loss_coef input allows you to weight the loss of each class differently. Say for example you have 2 classes, and the default loss coefficient gives you accuracies of 95% for label 0 and 70% for label 1, since label 0 has a lot more data points than label 1. You could use loss_coef = [1/n_0, 1/n_1], where n_0 and n_1 are the number of data points in each class respectively. This means more equal weight will be put on both classes, and could make your accuracies closer to 85% for both classes. 

Note that any value for loss_coef other than 1 will likely decrease your overall accuracy. 

See the examples section below for more information.

# predict() Function
Uses the k-means classifier to predict the classes of each given data point. 

Required Inputs: 
    x -> shape(n, m). n data points with m-dimensional coordinates.
    w -> The weight values calculated by the train() function.
    b -> The bias of the softmax calculated by the train() function.
    g -> The k-cluster centres calculated by the train() function.
        
Optional Inputs:
    y -> shape(n, ). n data points with class between 0 and v-1, where v is the 
            number of possible classes. If this is provided the function will print
            the accuracy of the model.

Outputs:
    l_out -> shape(n, ). The predicted labels for each data point.
    y_out -> shape(n, v). The certainty that each data point belongs to each class.
    
# Example 1: 2 dimensional space, 2 classes
This example uses data points gathered from a motion tracking camera. Each data point has a 2-dimensional position and an associated class that was identified manually by a user. A label of 1 means that the data point corresponds to a bicycle that passed by the camera, whereas a label of 0 means that the data point was not a bike (instead it might have been a car, a pedestrian, noise, etc.). The number of points with label == 0 is roughly 10,000, whereas the number of points with label == 1 is roughly 500.

Below is an example using the data and k = [30, 10]:
w, b, g, _, _ = train(x, y, k=[30,10])

Total testing accuracy: 97.56% <br>
Testing accuracy for label 0: 98.67% <br>
Testing accuracy for label 1: 75.26% <br>

<img src="images/k_means_classifier/example_0.png?raw=true"/>

Since there are more data points with label == 0, the training function skewed the results towards label 0 (notice that the testing accuracy for label 0 is nearly 100%, whereas the accuracy for label 1 is only 75%). 

If we are interested in finding a classifier that improves the accuracy of label 1, we could add a loss coefficient to weight each class equally by setting the optional parameter loss_coef to [1/n_0, 1/n_1], where n_0 and n_1 are the number of points with labels 0 and 1 respectively. 

Below is an example using loss coefficients weighted by population:
w, b, g, _, _ = train(x, y, k=[30,10], loss_coef=[1/np.sum(y==0),1/np.sum(y==1)])

Total testing accuracy: 96.1%
Testing accuracy for label 0: 95.92%
Testing accuracy for label 1: 100.0%

<img src="images/k_means_classifier/example_1.png?raw=true"/>

As you can see the overall accuracy was slightly impacted, but the accuracy of label 1 was significantly improved. 

# Example 2: 3 dimensional space, 5 classes
This example uses randomly generated data in a 3-dimensional space. A series of ellipsoids was generated, then a series of data points in a grid (each position between 0 and 1) was classified by whether or not they were in each ellipsoid, then the position of each point was randomly translated by ~N(0,0.03) in each dimension. k = [50, 20, 10, 20, 30] was selected based on visual inspection. 

The number of points in each class are as follows: [6876, 562, 94, 568, 1161]

Below is an example using the data and k = [50, 20, 10, 20, 30]:
w, b, g, _, _ = train(x, y, k=[50, 20, 10, 20, 30])

Total testing accuracy: 91.69%
Testing accuracy by label: [97.96, 61.03, 56., 67.88, 91.74]%

<img src="images/k_means_classifier/example_2.gif?raw=true"/>

It is obvious that the large classes (specifically classes 0 and 4) have significantly larger accuracies. In the animation above you can even see that the smaller classes (specifically class 2) have very little "real estate" in the decision boundary. 

If we are interested in improving the accuracy of each class, we can again weight each class by 1/n_i, where n_i is the number of data points in that class. 

Below is an example using loss coefficients weighted by population:
w, b, g, _, _ = train(x, y, k=[50,20,10,20,30], loss_coef=[1/np.sum(y==0),1/np.sum(y==1),1/np.sum(y==2),1/np.sum(y==3),1/np.sum(y==4)])

Total testing accuracy: 84.51%
Testing accuracy by label: [84.29, 90.08, 100., 79.82, 83.89]%

Below is a comparison between the default loss_coef and the adjusted loss coefficients as described above:
<img src="images/k_means_classifier/example_3.gif?raw=true"/>

As you can see, the overall accuracy went down, but the accuracies by label became much more equal. As you can see in the above animation the decision regions defined for the smaller classes became much large. See the decision region of class 2 for example: before adjusting the loss_coef there was only 1 decision region for class 2 and it was fairly small, whereas afterwards there are multiple regions for class 2 and they are much larger. Similarly, the decision region for class 0 became much smaller. 
