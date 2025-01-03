Title: The Street View House Numbers (SVHN) Cropped Dataset

Source: http://ufldl.stanford.edu/housenumbers/

SVHN is a real-world image dataest for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. 
It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled date (over 600000 digit images) 
and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). 
SVHN is obtained from house numbers in Google Street View Images. 
  
Number of Instances: over 600000 images, 73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data. The format is MNIST-like 32x32 images centered around a single character (many of the images do contain some distractors at the sides).

Attribute Information:
  Resnet18 network: The 512 attributes correspond to the second last layer of a ResNet18 model pretrained on ImageNet Data Set used to predict the class of each image on Svhn Cropped dataset. 
  Regnet_y_400mf network: The 440 attributes correspond to the second last layer of a Regnet_y_400mf model pretrained on ImageNet Data Set used to predict the class of each image on Svhn Cropped dataset. 
  Resnet34 network: The 512 attributes correspond to the second last layer of a ResNet34 model pretrained on ImageNet Data Set used to predict the class of each image on Svhn Cropped dataset. 
  Swin_v2_s network: The 768 attributes correspond to the second last layer of a Swin_v2_s model pretrained on ImageNet Data Set used to predict the class of each image on Svhn Cropped dataset. 
