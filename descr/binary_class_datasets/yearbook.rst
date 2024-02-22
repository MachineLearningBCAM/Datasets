Title: Yearbook Dataset

Sources: https://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html

Past Usage: 1. Ginosar, S., Rakelly, K., Sachs, S., Yin, B., & Efros, A. A. (2015).

A century of portraits: A visual historical record of american high school yearbooks. In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 1-7).

Kumar, A., Ma, T., & Liang, P. (2020, November). Understanding self-training for gradual domain adaptation. In International Conference on Machine Learning (pp. 5468-5479). PMLR.
Relevant Information: This part of the dataset contains 37,921 frontal-facing American high school yearbook portraits taken from 1905 to 2013.

Number of Instances: 37921 Class 0 (F) 20248 Class 1 (M) 17673

Number of Attributes: Grayscale PNG images

Attribute Information: 
    Resnet18 network: The 512 attributes correspond to the second last layer of a ResNet18  model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset. 
    Regnet_y_400mf network: The 440 attributes correspond to the second last layer of a Regnet_y_400mf model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset.
    Resnet34 network: The 512 attributes correspond to the second last layer of a ResNet34 model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset.
    Swin_v2_s network: The 768 attributes correspond to the second last layer of a Swin_v2_s model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset.
    ViT_B_16 network: The 1000 attributes correspond to the second last layer of a ViT_B_16 model pretrained on ImageNet Data Set used to predict the class of each image on Yearbook dataset.
    
Missing Attribute Values: None

This is a copy of Portraits Data Set. The original dataset can be download from this adress: https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0
