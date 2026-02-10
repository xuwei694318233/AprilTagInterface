import numpy as np

camera_matrix = np.array([
    [605.570481, 0.0, 323.994426],
    [0.0, 605.739564, 241.155806],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

dist_coeffs = np.zeros(5)  # RealSense畸变很小，填0即可

# 只保存必要的两个字段
np.savez('calibration/realsense_calib.npz', 
         mtx=camera_matrix, 
         dist=dist_coeffs)

print("标定文件创建成功")
print(f"内参: \n{camera_matrix}")
