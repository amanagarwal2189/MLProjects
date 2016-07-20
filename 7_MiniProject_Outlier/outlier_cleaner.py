#!/usr/bin/python

import operator
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        predictions is a list of predicted targets that come from your regression, 
        ages is the list of ages in the training set, 
        and net_worths is the actual value of the net worths in the training set
        
        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    #print("predictions",predictions)
    #print("net worth", net_worths)
    diff=(predictions-net_worths)**2
    #print("Diff", diff)
    cleaned_data = list(zip(ages, net_worths, diff))
    cleaned_data.sort(key=operator.itemgetter(2))
    cleaned_data=cleaned_data[:int(len(cleaned_data)*0.9)]
    #print(cleaned_data)
    '''
    My Code starts here
    '''
    

    '''
    My Code ends here
    '''

    
    return cleaned_data

