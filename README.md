# Dog-Classifier
My first Ai model. This repo includes the training script, the loading script to load a saved model and test it for yourself, and the model that I trained to show its accuracy after 10 epochs.


Data: the data directory is comprised of around 23,000 - 26,000 images of various animals. This data should be used for training your model.

Model: the model directory has only one trained model that is ready to go. It has an accuracy of ~90%.

trainModel.py: This script is what I came up with for training the model. While simple, it does the job efficiently and effectively. You can change the amount of epochs to train it more so that the training line and validation line get very close, making the model much closer to a 100% accuracy.

feedModel.py: This script is to be used on a trained and saved model. Using this, you can load up a model and feed it a single image of your choosing so you can see what decision is being made by the model and if it is correct.

Line Graph: The line graph that shows up after the training has completed will show the learning curve of the model. The blue line (test line) represents the data that the model learns from, while the orange line (validation line) represents the unseen data that the model validates its training on. Ideally, you want the two lines to be close together by the end of training. This means that the model not only learned from the training data, but can now confidently classify an image, even if it has not been seen before.

Lessons learned: With this being my first Ai model, I learned a lot about using tensorflow and how to train a model. I have already come up with some new ideas for my next project involving Ai, but next time I will be increasing the complexity a bit more to make an even more interesting set of data to interpret.

Notes: It is important that I really learn how a neural network works. While using a framework is very nice, I feel it would be quite beneficial to really learn the details of what goes on behind the scenes. Learning more about the neural network itself will allow me to make even more complex projects, maybe eventually I could even build my own neural network! For now, I will stick to frameworks and continue to experiment.
