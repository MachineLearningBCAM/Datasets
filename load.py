from os.path import join, dirname

import csv
import numpy as np
from sklearn.datasets.base import Bunch, load_files, load_data

def normalizeLabels(origY):
    """
    Normalize the labels of the instances in the range 0,...r-1 for r classes
    """

    # Map the values of Y from 0 to r-1
    domY = np.unique(origY)
    Y = np.zeros(origY.shape[0], dtype=int)

    for i, y in enumerate(domY):
        Y[origY == y] = i

    return Y

def load_adult(return_X_y=False):
    """Load and return the adult incomes prediction dataset (classification).

    =================   ==============
    Classes                          2
    Samples per class    [37155,11687]
    Samples total                48882
    Dimensionality                  14
    Features             int, positive
    =================   ==============

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    
    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary class datasets/', 'adult.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary class datasets/', 'adult.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file) # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.int)
    
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_diabetes(return_X_y=False):
    """Load and return the Pima Indians Diabetes dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               [500,268]
    Samples total                         668
    Dimensionality                          8
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/binary class datasets/', 'diabetes.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary class datasets/', 'diabetes.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_iris(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 3
    Samples per class              [50,50,50]
    Samples total                         150
    Dimensionality                          4
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/multi class datasets/', 'iris.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'iris.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype=np.int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype=np.int)
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_redwine(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                10
    Samples per class            [1599, 4898]
    Samples total                        6497
    Dimensionality                         11
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/multi class datasets/', 'redwine.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'redwine.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray([np.float(i) for i in d[:-1]], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_forestcov(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 7
    Samples per class [211840,283301,35754,
                     2747,9493,17367,20510,0]
    Samples total                      581012
    Dimensionality                         54
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi class datasets/', 'forestcov.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'forestcov.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file) # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.int)
    
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_letterrecog(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                26
    Samples total                       20000
    Dimensionality                         16
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi class datasets/', 'letter-recognition.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'letter-recognition.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file) # names of features
        feature_names = np.array(temp)

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[1:], dtype=np.float64)
            if d[0] in classes:
                index = classes.index(d[0])
                target[i] = np.asarray(index, dtype=np.int)
            else:
                classes.append(d[0])
                target[i] = np.asarray(classes.index(d[0]), dtype=np.int)
    
    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_ecoli(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 8
    Samples per class [143,77,52,35,20,5,2,2]
    Samples total                         336
    Dimensionality                          8
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/multi class datasets/', 'ecoli.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'ecoli.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp[1:])

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray([float(i) for i in d[1:-1]], dtype=np.float64)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype=np.int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype=np.int)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_vehicle(return_X_y=False):
    """Load and return the Iris Plants Dataset (classification).

    =================   =====================
    Classes                                 4
    Samples per class       [240,240,240,226]
    Samples total                         846
    Dimensionality                         18
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.
        
    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    
    fdescr_name = join(module_path, 'descr/multi class datasets/', 'vehicle.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'vehicle.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp[1:])

        classes = []
        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            if d[-1] in classes:
                index = classes.index(d[-1])
                target[i] = np.asarray(index, dtype=np.int)
            else:
                classes.append(d[-1])
                target[i] = np.asarray(classes.index(d[-1]), dtype=np.int)

    if return_X_y:
        return data, target

def load_segment(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 7
    Samples per class              [383, 307]
    Samples total                        2310
    Dimensionality                         19
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi class datasets/', 'segment.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'segment.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray([np.float(i) for i in d[:-1]], dtype=np.float64)
            except ValueError:
                print(i,d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_satellite(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 6
    Samples per class               383, 307]
    Samples total                        6435
    Dimensionality                         36
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi class datasets/', 'satellite.doc')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'satellite.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i,d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_optdigits(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                10
    Samples per class               383, 307]
    Samples total                        5620
    Dimensionality                         64
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/multi class datasets/', 'optdigits.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/multi class datasets/', 'optdigits.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i,d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_credit(return_X_y=False):
    """Load and return the Credit Approval prediction dataset (classification).

    =================   =====================
    Classes                                 2
    Samples per class               383, 307]
    Samples total                         690
    Dimensionality                         15
    Features             int, float, positive
    =================   =====================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary class datasets/', 'credit.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary class datasets/', 'credit.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int64)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            try:
                data[i] = np.asarray(d[:-1], dtype=np.float64)
            except ValueError:
                print(i,d[:-1])
            target[i] = np.asarray(d[-1], dtype=np.int64)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_magic(return_X_y=False):
    """Load and return the Magic Gamma Telescope dataset (classification).

    =========================================
    Classes                                 2
    Samples per class            [6688,12332]
    Samples total                       19020
    Dimensionality                         10
    Features                            float
    =========================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of adult csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr/binary class datasets/', 'magic.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data/binary class datasets/', 'magic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_glass(return_X_y=False):
    """Load and return the Glass Identification Data Set (classification).

    ===========================================
    Classes                                   6
    Samples per class    [70, 76, 17, 29, 13, 9]
    Samples total                           214
    Dimensionality                            9
    Features                              float
    ===========================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of glass csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'glass.csv')
    with open(join(module_path, 'descr', 'glass.rst')) as rst_file:
        fdescr = rst_file.read()

    data_file_name = join(module_path, 'data/binary class datasets/', 'glass.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=['RI: refractive index',
                                "Na: Sodium (unit measurement: "
                                "weight percent in corresponding oxide, "
                                "as are attributes 4-10)",
                                'Mg: Magnesium ',
                                'Al: Aluminim',
                                'Si: Silicon',
                                'K: Potassium',
                                'Ca: Calcium',
                                'Ba: Barium',
                                'Fe: Iron'],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_haberman(return_X_y=False):
    """Load and return the Haberman's Survival Data Set (classification).

    ==============================
    Classes                      2
    Samples per class    [225, 82]
    Samples total              306
    Dimensionality               3
    Features                   int
    ==============================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of haberman csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    with open(join(module_path, 'descr/binary class datasets/', 'haberman.rst')) as rst_file:
        fdescr = rst_file.read()

    data_file_name = join(module_path, 'data/binary class datasets/', 'haberman.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=['PatientAge', 
                                'OperationYear', 
                                'PositiveAxillaryNodesDetected'],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_mammographic(return_X_y=False):
    """Load and return the Mammographic Mass Data Set (classification).

    ==============================
    Classes                      2
    Samples per class    [516, 445]
    Samples total              961
    Dimensionality               5
    Features                   int
    ==============================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of mammographic csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    with open(join(module_path, 'descr/binary class datasets/', 'mammographic.rst')) as rst_file:
        fdescr = rst_file.read()

    data_file_name = join(module_path, 'data/binary class datasets/', 'mammographic.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.str)
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.str)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=['BI-RADS',
                                'age',
                                'shape',
                                'margin',
                                'density'],
                 DESCR=descr_text,
                 filename=data_file_name)

def load_indian_liver(return_X_y=False):
    """Load and return the Indian Liver Patient Data Set 
    (classification).

    =========================================================
    Classes                                                 2
    Samples per class                              [416, 167]
    Samples total                                         583
    Dimensionality                                         10
    Features                                       int, float
    Missing Values                                     4 (nan)
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)
    """data, target, target_names = load_data(module_path,
                                           'indianLiverPatient.csv')"""
    with open(join(module_path, 'data/binary class datasets/',
                   'indianLiverPatient.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)
    with open(join(module_path, 'descr/binary datasets/',
                   'indianLiverPatient.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['Age of the patient',
                                'Gender of the patient',
                                'Total Bilirubin',
                                'Direct Bilirubin',
                                'Alkaline Phosphotase',
                                'Alamine Aminotransferase',
                                'Aspartate Aminotransferase',
                                'Total Protiens',
                                'Albumin',
                                'A/G Ratio'])

def load_heart(return_X_y=False):
    """Load and return the Heart Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                         270
    Dimensionality                                         13
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data',
                   'heart.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(270)
        n_features = int(13)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])

def load_sonar(return_X_y=False):
    """Load and return the Sonar Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                         208
    Dimensionality                                         60
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data',
                   'sonar.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(208)
        n_features = int(60)
        data = []
        target = []

        for i, ir in enumerate(data_file):
            # print(float(ir[-1]))
            if len(ir[1:]) == 60:
                data.append(np.asarray(ir[1:], dtype=np.float64))
                target.append(int(ir[0]))
        data = np.asarray(data)
        target = np.asarray(target)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])

def load_svmguide3(return_X_y=False):
    """Load and return the SVM guide Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                        1243
    Dimensionality                                         21
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data',
                   'svmguide3.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(1243)
        n_features = int(21)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[1:22], dtype=np.float64)
            target[i] = np.asarray(ir[0], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])

def load_liver_disorder(return_X_y=False):
    """Load and return the Liver Disorder Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                         345
    Dimensionality                                          5
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data',
                   'liver_disorder.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(345)
        n_features = int(5)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(int(float(ir[-1])), dtype=np.float64)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])

def load_german_numer(return_X_y=False):
    """Load and return the German numer Data Set
    (classification).

    =========================================================
    Classes                                                 2
    Samples total                                        1000
    Dimensionality                                         24
    Features                                       int, float
    =========================================================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of satellite csv dataset.

    (data, target) : tuple if ``return_X_y`` is True

    """
    module_path = dirname(__file__)

    with open(join(module_path, 'data',
                   'german_numer.csv')) as csv_file:
        data_file = csv.reader(csv_file)
        target_names = []
        n_samples = int(1000)
        n_features = int(24)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples, ), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[1:], dtype=np.float64)
            target[i] = np.asarray(ir[0], dtype=int)

    trans = SimpleImputer(strategy='median')
    data = trans.fit_transform(data)

    if return_X_y:
        return data, normalizeLabels(target)

    return Bunch(data=data, target=normalizeLabels(target),
                 target_names=target_names,
                 DESCR=None,
                 feature_names=[])