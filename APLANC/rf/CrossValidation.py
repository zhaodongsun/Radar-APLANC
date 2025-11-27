from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

est = np.load('nndl/rf/final_array_est.npy')
gt = np.load('nndl/rf/final_array_gt.npy')
print(est)
print(gt)
mae = np.mean(np.abs(gt-est))
rmse = np.sqrt(np.mean((gt-est) ** 2))
r,_ = stats.pearsonr(gt,est)
print(f'mae:{mae}')
print(f'rmse:{rmse}')
print(f'r:{r}')

ground_truth = gt  
predictions = est  

plt.figure(figsize=(10, 8))
plt.scatter(ground_truth, predictions, color='blue', alpha=0.6, label='Data Points')

  
max_val = max(max(ground_truth), max(predictions)) + 10
plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Prediction (y=x)')

plt.title('Ground Truth vs Predicted Heart Rate', fontsize=14)
plt.xlabel('Ground Truth HR (BPM)', fontsize=12)
plt.ylabel('Predicted HR (BPM)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()