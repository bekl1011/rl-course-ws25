# Lab 5

In this lab we will look into the world of deep learning and predict on the
[Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) using PyTorch.


### Task 1:
Install [PyTorch](https://pytorch.org/get-started/locally/) on your local machine
(or try out [Google Colab](https://colab.research.google.com/)).

### Task 2:
Have a look at the notebook [`5_PyTorch_Example.ipynb`](5_PyTorch_Example.ipynb).
1. Go through the notebook and try to understand what is going on.
2. Make this code run on your machine (or on Google Colab).
2. Play around with the code and observe the effects on the accuracy of the test data:
    - Introduce a second hidden layer with size 5.
    - Change the learning rate to a very low value and a very high value. What is the effect on the training loss?
    - Train a smaller or bigger number of epochs.

### Task 3:
Make a prediction with the trained network on the data point: `[4.9, 3.0, 1.4, 0.2]`.
What are the probabilities for the different classes?
Hint: Since the network outputs raw scores, you need to apply the [softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
function to normalize the network's output to probabilities.
