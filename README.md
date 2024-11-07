# Datasets
This repository contains some of the common dataset that are used. The datasets available here are - 

## Datasets with only two classes

1) Adult (adult.csv)
2) Credit (credit.csv)
3) Diabetes (diabetes.csv)
4) Haberman (haberman.csv)
5) Indian Liver Patient (indianLiverPatient.csv)
6) Magic (magic.csv)
7) Mammographic (mammographic.csv)
8) Pulsar (pulsar.csv) - Only data
9) Heart Diseases (heart.csv)
10) Connectionist Bench (Sonar, Mines vs. Rocks) (sonar.csv)
11) SVM guide (svmguide3.csv)
12) Liver Disorder (liver_disorder.csv)
13) German credit data (german_numer.csv)
14) Yearbook dataset (Yearbook/portraits_1905_1954.mat, portraits_1955_1974.mat, portraits_1975_1994.mat, portraits_1995_2013.mat)

## Datasets with multiple classes

1) Ecoli (ecoli.csv)
2) Forest Cover (forestcov.csv)
3) Glass (glass.csv)
4) Iris (iris.csv)
5) Letter Recognition (letter-recognition.csv)
6) Optical Digit Recognition (optdigits.csv)
7) Wine Quality (redwine.csv)
8) Satellite (satellite.csv)
9) Image Segmentation (segment.csv)
10) Vehicle Silhouettes (vehicle.csv)
11) Pulsar (pulsar.csv) - Only data
12) DomainNet (domain_4_clases.mat) The “DomainNet” dataset contains six different domains with decreasing realism and the
goal is to predict if an image is an airplane, bus, ambulance, or police car. The sequence
of tasks corresponds to the six domains: real, painting, infograph, clipart, sketch, and
quickdraw.

The datasets are in the [data](https://github.com/MachineLearningBCAM/Datasets/tree/main/data) folder and their description is available in the folder [descr](https://github.com/MachineLearningBCAM/Datasets/tree/main/descr)

# Example 

The repo also contains some functions in the file `load.py` to load these datasets as a numpy matrix. The file `example.py` gives an example of the usage of these functions. You can run that file to load and see the output of any of these datasets by passing the name of the dataset file as the command line argument - 

```
python example.py datasetname
```

In order to load a dataset, you can call the corresponding function (`load_<datasetname>`) available in the file `load.py`. For example, to load the dataset adult, you need to call the function `load_adult(True)`. Note: you need to pass `True` as parameter to the function if you want the function to return the dataset and its labels as numpy matrix and vector respectively.


## Reference

These datasets are taken from the UCI machine learning [repository](https://archive.ics.uci.edu/ml/datasets.php) and LIBSVM Data [repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)

