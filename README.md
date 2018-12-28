Objective : Search for the anomaly in click stream data given.

According to my knowledge, we can apply different approaches and algorithms to identify different types of anomalies as follows.

Collective Anomalies : Cluster, Isolation Forest, One Class SVM
Contextual Anomalies : Cluster + Point Anomaly Methods
Point Anomalies : Elliptic Envelope
Ordered Anomalies : Markov Chain, RNN
Collective Ordered anomalies : cluster + markov chain

Since, we are not given any output variable and also we don't have specific information to find the output variable. This is case of Unsupervised Learning. We also don't have information about which type of anomalies the given data have.

Therefore, I have checked for anomalies in the data for 5 Algorithms. 
- Since there is high possibility that the anomalies can be collective, I have taken K-means clustering, Isolation Forest and One Class SVM into consideration. 
- If collective anomalies are of type point categorical, then I have used Elliptic Envelope.
- To check Ordered anomalies, I have used RNN. RNN will predict one feature based on rest features. And if prediction is highly erronious in some points, then it will consider them as anomalies.


HOW TO RUN ?
- I have developed my whole project into single Jupyter notebook. Also, I have written markdown instructions wherever needed so that it won't be confusing at all.
- functions.py file is not to execute. It only contains methods, which are used in notebook. 
- All other files except jupyter notebook and functions.py are .csv files. These files are already provided data.
- Directly you can open jupyter notebook and execute cells in sequential manner.