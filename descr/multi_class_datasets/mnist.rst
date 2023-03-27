Title: Mnist Dataset

Sources: https://www.tensorflow.org/datasets/catalog/mnist?hl=es-419


The MNIST database of handwritten digits, available from this page, has a training set of 60000 examples, and a test set of 10000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

Four files are available:
train-images-idx3-ubyte.gz: training set images (9912422 bytes)
train-labels-idx1-ubyte.gz: training set labels (28881 bytes)
t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)
t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)

Source code: tfds.image_classification.MNIST

Homepage: http://yann.lecun.com/exdb/mnist/

Number of Instances: 60000 train images, 10000 test images

Attribute Information: 
    Resnet18 network: The 512 attributes correspond to the second last layer of a ResNet18  model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset. 
    Regnet_y_400mf network: The 440 attributes correspond to the second last layer of a Regnet_y_400mf model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset.
    Resnet34 network: The 512 attributes correspond to the second last layer of a ResNet34 model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset.
    Swin_v2_s network: The 768 attributes correspond to the second last layer of a Swin_v2_s model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset.
    
