# TPGraph
Figure 1: Dynamic Spatial-Temporal Traffic Prediction with Attentional Graph Fusion: A Case Study on Arterial Roads
![image](https://user-images.githubusercontent.com/73110791/188844885-152245b6-7c67-4b07-bcf3-28e77ce10e16.png)

*The motivation of each layer: we define traffic prediction as a temporal- and spatial- dependencies problem. The temporal and spatial characteristics inside the data should be efficiently utilized. Firstly, considering that traffic data is recorded for up to several months or years, much historical information of days and weeks could provide experience for the model to learn, so we have introduced a multi-scale temporal feature fusion to integrate the historical information as shown in Figure. Subsequently, to introduce the spatial dependencies into the model, we have designed a multi-graph convolution module to extract the spatial feature of the road map. Following, to balance, integrate and fully learn the fused feature, we have proposed a dynamic spatial-temporal prediction module to implement the final learning and predicting process. Finally, to make our model scalable meanwhile balance the accuracy and the computation cost, we have reconstructed our modules to get the final pipeline as shown in the Figure.

## Code running  
We have conducted the short-term experiments on Aliyun dataset as the preliminary experiment, and long-term experiments on Aliyun and Pems-bay datasets as the main experiments.
Here, we provide a code for long-term prediction on Pems-bay dataset. //
Firstly, please unzip the pems-bay file.//
The model file is TPGraph.py. You can run the code by running the main_pems.py. Meanwhile, our best result is recorded in BEST_REC.

## Running example on  Aliyun dataset
**Figure 2  A running example with executing input and output of stages.
![image](https://user-images.githubusercontent.com/73110791/189161478-94b68228-e3b1-4c49-b1bd-3670a41d264c.png)


Layer 1	
Input: Raw traffic data D [#REC, N] = [29640,100];	
Output: Temporal Feature TF: [B, C, N, T] = [16, 128, 100, 12]	
	
	
Given raw traffic data D, Layer 1 captures its recently-, daily-, and weekly-periodic (temporal) features. For example, the multi-scale temporal features fusion module collects 16 batches of the recent 8 minutes data, recent 2 days data, and recent 2 weeks data to form the xr [16, 100, 8], xd [16, 100, 2], and xw [16, 100, 2]. 	
	
Layer 2	
Input: Temporal Feature TF (Layer 1's output), Road network topology G: [N, N]	
Output: Spatial Feature SF [B, C, N, T] = [16, 128, 100, 12]	
	
	
Using the Temporal Feature TF (from Layer 1) and Road network topology G, Layer 2 extracts the spatial features (e.g. topological structure and traffic correlation) and ouput SF [B, C, N, T] = [16, 128, 100, 12]	
	
	
Layer 3 	
Input: Spatial Feature SF (Layer 2's output), Temporal Feature TF (Layer 1's output); 	
Output: result: [B, N, T_O] = [16, 100, 3]	
	
	
Layer 3 integrates the spatial and temporal features from Layer 1 and Layer 2, and outputs the prediction results [B, N, T_O] = [16, 100, 3], where 3 corresponds to the next 3 time slices in the future.	
	
	
Pipeline	
input: the input of L1	
output: the output of L3	

***********
tij: average travel time in road i at time j


#REC - number of recorded data	


B - batch size	


C - embed size	


N - number of roads 	


T - data length	


T_O - future time step	



## Dataset and preliminary experiments
Preliminary exp. run in Aliyun dataset

**Aliyun dataset** link: https://tianchi.aliyun.com/competition/entrance/231598/information?from=oldUrl

1. Our ablation experiment in Table 3 did not report the integration of each module, making some results counterintuitive (e.g. abnormal MAPE). Here, we provide a NEW ablation experiment table of each modules with Aliyun dataset shows in Table A.
<!-- ![image](https://user-images.githubusercontent.com/73110791/188572746-4d7e5815-ab91-4cef-acf8-649b3a2e36eb.png) -->
![image](https://user-images.githubusercontent.com/73110791/189019542-feadfa4b-3062-4378-b8c5-fe513bd2e91c.png)


*In task 1, without multi-scale temporal features fusion, the accuracy error increases 15.6%; without temporal fusion and temporal transformer, the accuracy error increases 38.7%; without multi-graph convolution and spatial transformer, the accuracy error increases 42.1%.

We also split our model into two tracks of temporal and spatial to compared with some baselines.

In temporal track, we turn off the spatial extracting part of the final pipeline shows in Table B.
![image](https://user-images.githubusercontent.com/73110791/188601395-49e61831-7dbf-4ce7-acc0-8fd0ebf9bd91.png)
 We have compared with LSTM, GRU, and LSTM_BILSTM with our temporal track components and the results show that our temporal parts achieve prediction accuracy with only a little compromise in compuational time.


In spatial track, we turn off the temporal extracting part of the final pipeline shows in Table C.
![image](https://user-images.githubusercontent.com/73110791/188572649-2d4ae6f6-3031-4735-8d34-a73f04ec25c1.png) 
 We have compared with GCN, TGVN, and STGCN with our spatial track components and the results show that our spatial parts outperform them in accuracy and compuational time.

Table D: Comparsion with all the baselines and the final pipeline TPGraph
![image](https://user-images.githubusercontent.com/73110791/188990627-7992c377-4b2a-48f9-87b7-5fa3baad378c.png)

The final pipeline achieves almost 27% prediction accuracy with only a little compromise in compuational time, and can scale well and easily to be paralelisable on million-sized graphs

2. Meanwhile, some preliminary experiments with basic datasets were added to ensure the scalability of our model. More experiments of the followed two datasets will be implemented in the future:

1) Road network of California (1,965,206 nodes, and 2,766,607 edges): https://snap.stanford.edu/data/roadNet-CA.txt.gz
2) Road network of Pennsylvania (1,088,092 nodes, and 1,541,898 edges)：https://snap.stanford.edu/data/roadNet-PA.txt.gz

After partition of the road network of California with 2,766,607 edges, each class contains 2000~ edges and the total number of observed traffic data points is about 6,000,000. In our new experiment, the training time of each epoch is 13.37s and the simple tested MAE and RMSE are 6.8295 and 12.4544., highlighting the scalability and practicality of our scheme on large datasets.

To further ensuring that our framework has scalablity and practicality. We selected part of roads of the class to tested the correlation between training time and edges. Per epoch training time of the partitioned class are 5.73 s, 9.29 s, 13.37 s, and 25.10 s corresponding to the number of edges are 400, 800, 1200, and 1722. The results show that our training time increase is linear.

**Figure 3: The computational time cost of scability 

![image](https://user-images.githubusercontent.com/73110791/189132658-e1235562-feaa-4376-a955-46099eac9aab.png)

Some datasets are utilized to implement this part:

Metr-LA dataset (This dataset contains 207 nodes and 1722 edges. The total number of observed traffic data points is 6,519,002.)

PEMS-BAY dataset (This dataset contains 325 nodes and 2694 edges. The total number of observed traffic data points is 16,937,179.)

<!-- **Metr-LA dataset** (This traffic dataset contains traffic information collected from loop detectors in the highway of Los Angeles County. We select 207 sensors and collect 4 months of data ranging from Mar 1st 2012 to Jun 30th 2012 for the experiment. This dataset contains 207 nodes and 1722 edges. The total number of observed traffic data points is 6,519,002.):

*We have run 10 epoch for testing the feasibility to utilize our model with non-processed data of Metr-LA, the results show that the learned edges increased, the training time increased in nearly linear. The per epoch training time of  5.73 s, 9.29 s, 13.37 s, and 25.10 s corresponding to the number of edges are 400, 800, 1200, and 1722. The 10 epoch MAE and RMSE are 6.8295 and 12.4544.

**PEMS-BAY dataset** (This traffic dataset is collected by California Transportation Agencies (Cal-Trans) Performance Measurement System (PeMS). We select 325 sensors in the Bay Area and collect 6 months of data ranging from Jan 1st 2017 to May 31th 2017 for the experiment. This dataset contains 325 nodes and 2694 edges. The total number of observed traffic data points is 16,937,179.):

*Also, we have utilized the non-processed data of PEMS-BAY to gain the preliminary results: The per epoch training time of 9.32 s, 23.21 s, 75.07 s corresponding to the number of edges are 400, 1000, and 2694. The 10 epoch MAE and RMSE are 11.443 and 18.804.

**Metr-LA dataset and PEMS-BAY dataset** link：https://pan.baidu.com/s/1w6RmViZWFsxAKouwhx32uQ?pwd=6666 code：6666  -->

Long-term prediction for 1 hour has also been conducted on Aliyun and PeMs-bay datasets, the detailed results can be observed in our paper.
