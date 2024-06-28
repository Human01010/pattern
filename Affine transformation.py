import cv2
import numpy as np
# 读取输入图像
input_image = cv2.imread('C:\pyprojects\lessons\pattern_detection\Project\char_tmpl/9.jpg')
# 定义仿射变换的源和目标点
# 这里定义了一个简单的仿射变换
rows, cols = input_image.shape[:2]
src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
dst_points = np.float32([[0, rows * 0.3], [cols * 0.85, rows * 0.25], [cols * 0.15, rows - 1]])
# 计算仿射变换矩阵
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
# 进行仿射变换
output_image = cv2.warpAffine(input_image, affine_matrix, (cols, rows))
# 保存结果到文件
cv2.imwrite('C:\pyprojects\lessons\pattern_detection\Project\char_tmpl/29.jpg', output_image)
# 显示原始图像和变换后的图像（可选）
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
