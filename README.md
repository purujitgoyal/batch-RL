### BATCH-RL HCOPE

- Make sure latest version of Anaconda is installed for python 3.7.
- run 
    ``pip install cma``
    to install cma-es library
- Data pre processing has been done in 'Data Preprocessing' notebook and histories data has been stored
in a new 'histories.csv', while all other metadata like num of actions, state features, confidence have been 
hardcoded.
- run ``nohup python hcope.py &`` to run the code. It will save the logs to 'nohup.out' file while candidate thetas
which pass the safety test will get saved to i.csv files.
