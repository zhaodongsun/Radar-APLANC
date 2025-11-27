# Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast
This is the official code repository of our paper "Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast".
The method does not require any ground truth for training.
## Prerequisite
Please check `requirement.txt` for the required Python libraries.
## Dataset and Pre-prep
Our [RHB](https://1drv.ms/u/c/0591660FD4399B1C/ET_1hR00cFROj6Bt1PxA2PEBA7NojefujFp8M5vvLLvXnw?e=gKOuxT) dataset maintains the same data format as the Equipleth dataset and is processed using `organizer.py`. If you want to use your own dataset or other datasets, please replace `organizer.py`.

Hierarchy of the RHB dataset
```
dataset
|--- RHB
|        |
|        |--- 1_1(volunteer id 1 trial 1) 
|        |         |
|        |         |--- rf.pkl(Radar data)
|        |         |--- vital_dict.npy(ground truth ppg)
|        |
|        |
|        |--- 1_2(volunteer id 1 trial 2) 
|        |
|        |
|        |
|        |--- 1_3(volunteer id 1 trial 3) 
|        |
|        |
|        |
|        |--- 2_1(volunteer id 2 trial 1) 
|        |
|        |
|        |
|
|
|--- RHB_demo_fold1.pkl(folds pickle file)
|--- RHB_demo_fold2.pkl(folds pickle file)
|--- RHB_demo_fold3.pkl(folds pickle file)
|--- RHB_demo_fold4.pkl(folds pickle file)
```
RHB_demo_fold[index].pkl files are the data partitions we use for cross validation. RHB_Pseudo_Generate.py is used to generate Augumented pseudo labels after the first stage. You need to ensure that the fold_math in RHB_Pseudo_Generate.py is the same as that in the first stage train.py. You can train each fold by changing the fold_path in the train.py file. If you want to test the model, please change the fold_path in test.py after each fold training and run it. This will generate a temp_est.json and temp_ get.json file to store the predicted and true values of the test set for this fold model. After testing all four fold data, run CrossValData_combine.py and CrossValidation.py for cross validation testing.
## Execution

### Training
Please make sure your dataset is processed as described above. You only need to modify the code in a few places in `train.py` and `eval.py` to start your training.  After modifying the code, you can directly run
```
python train.py
```
### Testing
After training, you can test the model on the test set. You can directly run
```
python test.py
```
