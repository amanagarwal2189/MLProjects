#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "br") )

features = ["salary", "bonus"]
"""
Remove TOTAL outlier
"""
data_dict.pop("TOTAL", 0)

"""
Removed 1 major outlier
"""
e={k:v for k,v in data_dict.items() if v['salary'] != 'NaN' and int(v['salary'])>=1000000 and v['bonus']!='Nan' and int(v['bonus'])>=5000000}
print ({k for k,v in e.items()})
data = featureFormat(data_dict, features)

### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


