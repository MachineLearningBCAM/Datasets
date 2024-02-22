Title: Cifar10 Dataset

Sources: https://www.tensorflow.org/datasets/catalog/cifar10?hl=es-419


The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Source code: tfds.image_classification.Cifar10

Homepage: https://www.cs.toronto.edu/~kriz/cifar.html

Number of Instances: 50000 train images, 10000 test images

Attribute Information: 
    Resnet18 network: The 512 attributes correspond to the second last layer of a ResNet18  model pretrained on ImageNet Data Set used to predict the class of each image on Cifar10 dataset. 
    Regnet_y_400mf network: The 440 attributes correspond to the second last layer of a Regnet_y_400mf model pretrained on ImageNet Data Set used to predict the class of each image on Cifar10 dataset.
    Resnet34 network: The 512 attributes correspond to the second last layer of a ResNet34 model pretrained on ImageNet Data Set used to predict the class of each image on Cifar10 dataset.
    Swin_v2_s network: The 768 attributes correspond to the second last layer of a Swin_v2_s model pretrained on ImageNet Data Set used to predict the class of each image on Cifar10 dataset.
    ViT_B_16 network: The 768 attributes correspond to the second last layer of a ViT_B_16 model pretrained on ImageNet Data Set used to predict the class of each image on Cifar10 dataset.
    
